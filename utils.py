"""Small stateless helpers for text cleanup and video frame encoding."""
import re
import cv2
import base64
from config import GOOGLE_SPEECH_LANGUAGES

def clean_string(text: str) -> str:
    """Strips markdown (**bold**, *italic*, # headings) so TTS doesn't read the symbols aloud."""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'#(.*?)\n', r'\1\n', text)
    return text.strip()

def encode_and_resize_frame(frame, max_dim: int, quality: int = 90) -> str:
    """Downscales an OpenCV BGR frame to max_dim on its longest side and JPEG-encodes it as base64."""
    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode('utf-8')

def get_speech_lang_code(lang_name: str) -> str:
    """Maps a display language name (e.g. "French") to its BCP-47 code, defaulting to en-US."""
    return GOOGLE_SPEECH_LANGUAGES.get(lang_name, "en-US")