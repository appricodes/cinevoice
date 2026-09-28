"""Small stateless helpers for text cleanup and video frame encoding."""
import re
import base64
from config import GOOGLE_SPEECH_LANGUAGES

def clean_string(text: str) -> str:
    """Strips markdown (**bold**, *italic*, # headings) so TTS doesn't read the symbols aloud."""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'#(.*?)\n', r'\1\n', text)
    return text.strip()

def resize_to_max_dim(frame, max_dim: int):
    """Downscales an OpenCV BGR image to max_dim on its longest side; a no-op if it already fits."""
    # Imported here, not at module scope: this module is pulled in by the main window, so a
    # top-level import would load OpenCV during startup for every user, including the many
    # who never capture a frame. vision_manager and local_vlm defer cv2 for the same reason.
    import cv2
    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
    return frame

def encode_frame(frame, quality: int = 90) -> str:
    """JPEG-encodes an OpenCV BGR image as base64, with no resizing."""
    import cv2
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode('utf-8')

def encode_and_resize_frame(frame, max_dim: int, quality: int = 90) -> str:
    """Downscales an OpenCV BGR frame to max_dim on its longest side and JPEG-encodes it as base64."""
    return encode_frame(resize_to_max_dim(frame, max_dim), quality)

def build_frame_grid(frames: list):
    """
    Tiles up to nine OpenCV BGR frames into one composite image, ordered left-to-right then
    top-to-bottom (row-major, oldest frame first), using as many of 3 columns as the frame
    count allows. Frames narrower or shorter than the largest one in the group are placed in
    the top-left of their cell over black, so every cell still lines up on a shared grid.
    """
    import numpy as np
    n = len(frames)
    cols = min(3, n)
    rows = -(-n // cols)  # ceil division
    cell_h = max(f.shape[0] for f in frames)
    cell_w = max(f.shape[1] for f in frames)
    canvas = np.zeros((cell_h * rows, cell_w * cols, 3), dtype=np.uint8)
    for idx, frame in enumerate(frames):
        r, c = divmod(idx, cols)
        h, w = frame.shape[:2]
        canvas[r * cell_h:r * cell_h + h, c * cell_w:c * cell_w + w] = frame
    return canvas

def build_frame_grids(frames: list, group_size: int = 9) -> list:
    """
    Chunks frames into groups of at most group_size (oldest first) and tiles each group into
    one composite image via build_frame_grid, so e.g. 90 frames become 10 images of 9 tiles
    each. A leftover group of exactly one frame is passed through unchanged rather than
    "tiled" into a pointless single-cell canvas.
    """
    grids = []
    for i in range(0, len(frames), group_size):
        group = frames[i:i + group_size]
        grids.append(group[0] if len(group) == 1 else build_frame_grid(group))
    return grids

def get_speech_lang_code(lang_name: str) -> str:
    """Maps a display language name (e.g. "French") to its BCP-47 code, defaulting to en-US."""
    return GOOGLE_SPEECH_LANGUAGES.get(lang_name, "en-US")