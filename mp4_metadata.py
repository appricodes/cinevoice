"""
Reads and writes Cinevoice's AI-generated descriptions (the full-video story and the
30-second block narrations) directly inside an MP4/MOV file, as a top-level 'uuid' box --
the ISO/IEC 14496-12 mechanism reserved for private, application-specific data.

This box is always appended after every existing box (ftyp/moov/mdat/...) and is never
inserted between them, so nothing already in the file ever moves: sample tables and byte
offsets inside 'moov' stay valid, and any player -- including this app's own Qt/ffmpeg
backend -- that walks top-level boxes by size and skips types it doesn't recognize will
simply ignore ours.

Deliberately dependency-free: no ffmpeg subprocess and no mutagen. This project's
THIRD-PARTY-LICENSES.txt documents removing ffmpeg specifically to avoid GPL obligations,
and mutagen is GPLv2, so it would reintroduce the same problem. Plain struct/json instead.
"""
import json
import os
import struct

_BOX_TYPE = b'uuid'
_USERTYPE = b"CineVoiceMDataV1"  # 16-byte extended_type identifying our box specifically
_MAGIC = b'CVMD1'  # payload prefix + format version, for forward compatibility

class _UnsupportedLayout(Exception):
    """The file's last box has an open-ended (extends-to-EOF) size -- e.g. a still-growing
    mdat some capture tools emit. Appending after it would require shifting that box's
    declared size across an offset boundary other tools may rely on, so we decline rather
    than risk it; the caller falls back to a sidecar file for a video shaped like this."""

def _scan_boxes(f, file_size):
    pos = 0
    while pos + 8 <= file_size:
        f.seek(pos)
        header = f.read(8)
        if len(header) < 8:
            return
        size32, box_type = struct.unpack('>I4s', header)
        if size32 == 1:
            largesize_bytes = f.read(8)
            if len(largesize_bytes) < 8:
                return
            size = struct.unpack('>Q', largesize_bytes)[0]
            header_len = 16
        elif size32 == 0:
            raise _UnsupportedLayout()
        else:
            size = size32
            header_len = 8
        if size < header_len or pos + size > file_size:
            return
        yield pos, header_len, box_type, size
        pos += size

def _find_our_box(f):
    """Returns (box_start, box_end, payload_start) for our uuid box, or None if absent."""
    f.seek(0, os.SEEK_END)
    file_size = f.tell()
    for pos, header_len, box_type, size in _scan_boxes(f, file_size):
        if box_type == _BOX_TYPE:
            f.seek(pos + header_len)
            if f.read(16) == _USERTYPE:
                return pos, pos + size, pos + header_len + 16
    return None

def read_metadata(path: str) -> dict:
    """Reads Cinevoice's embedded story/block descriptions from path; {} if none present."""
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, 'rb') as f:
            found = _find_our_box(f)
            if not found:
                return {}
            box_start, box_end, payload_start = found
            f.seek(payload_start)
            payload = f.read(box_end - payload_start)
    except (_UnsupportedLayout, OSError):
        return {}
    if not payload.startswith(_MAGIC):
        return {}
    try:
        return json.loads(payload[len(_MAGIC):].decode('utf-8'))
    except (ValueError, UnicodeDecodeError):
        return {}

def write_metadata(path: str, data: dict) -> bool:
    """
    Replaces Cinevoice's embedded uuid box (or appends one if absent) with data as JSON.
    Safe by construction: existing boxes are never rewritten in place, only re-appended at
    the same tail position, so nothing before them ever shifts.

    The new box is written before the file is truncated, never after. Truncating first would
    mean a crash in between could take anything that followed our box with it -- the user's
    own video data -- and this writes into files the user cannot replace.
    """
    if not path or not os.path.exists(path):
        return False
    payload = _MAGIC + json.dumps(data, ensure_ascii=False).encode('utf-8')
    box_size = 8 + 16 + len(payload)
    if box_size < 2 ** 32:
        header = struct.pack('>I4s', box_size, _BOX_TYPE)
    else:
        header = struct.pack('>I4s', 1, _BOX_TYPE) + struct.pack('>Q', box_size + 8)
    new_box = header + _USERTYPE + payload
    try:
        with open(path, 'r+b') as f:
            found = _find_our_box(f)
            trailing = b''
            if found:
                box_start, box_end, _ = found
                f.seek(box_end)
                trailing = f.read()
                f.seek(box_start)
            else:
                f.seek(0, os.SEEK_END)
            f.write(new_box)
            if trailing:
                f.write(trailing)
            # Only now drop whatever the old, longer box left behind past the new end.
            f.truncate(f.tell())
            f.flush()
            os.fsync(f.fileno())
        return True
    except (_UnsupportedLayout, OSError):
        return False
