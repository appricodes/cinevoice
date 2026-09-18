"""Static configuration: AI model catalog, tunable constants, and language tables."""
import os
import sys

MODELS = [
{
       "model_name": "Grok 4.3 - no reasoning",
       "provider_name": "xAI",
       "model_id": "grok-4.3",
       "provider_id": "grok",
       "endpoint": "https://api.x.ai/v1/chat/completions",
       "parameters": {
          "reasoning_effort": "none"
       }
    },
    {
       "model_name": "Grok 4.6",
       "provider_name": "xAI",
       "model_id": "grok-4.6",
       "provider_id": "grok",
       "endpoint": "https://api.x.ai/v1/chat/completions",
       "parameters": {
          "reasoning_effort": "low"
       }
    },
    {
       "model_name": "Gemini 2.5 Flash Lite",
       "provider_name": "Google",
       "model_id": "gemini-2.5-flash-lite",
       "provider_id": "gemini",
       "endpoint": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
       "parameters": {}
    },
    {
       "model_name": "Gemini 3.1 Flash Lite",
       "provider_name": "Google",
       "model_id": "gemini-3.1-flash-lite",
       "provider_id": "gemini",
       "endpoint": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
       "parameters": {}
    },
    {
       "model_name": "GPT-4o Mini",
       "provider_name": "OpenAI",
       "model_id": "gpt-4o-mini",
       "provider_id": "openai",
       "endpoint": "https://api.openai.com/v1/chat/completions",
       "parameters": {}
    },
    {
       "model_name": "GPT-5 Nano",
       "provider_name": "OpenAI",
       "model_id": "gpt-5-nano",
       "provider_id": "openai",
       "endpoint": "https://api.openai.com/v1/chat/completions",
       "parameters": {}
    },
    {
       # No "reasoning_effort" on any Mistral entry: the API rejects the field outright on
       # models that do not reason (only the magistral family does), answering 422 with
       # "Reasoning effort can not be set for this model" instead of ignoring it the way
       # xAI and OpenAI do. Sending it here breaks every generation with these models.
       "model_name": "Mistral ministral 3b",
       "provider_name": "Mistral",
       "model_id": "ministral-3b-latest",
       "provider_id": "mistral",
       "endpoint": "https://api.mistral.ai/v1/chat/completions",
       "parameters": {}
    },
    {
       "model_name": "Mistral ministral 14b",
       "provider_name": "Mistral",
       "model_id": "ministral-14b-latest",
       "provider_id": "mistral",
       "endpoint": "https://api.mistral.ai/v1/chat/completions",
       "parameters": {}
    },
    {
       "model_name": "Mistral Small",
       "provider_name": "Mistral",
       "model_id": "mistral-small-latest",
       "provider_id": "mistral",
       "endpoint": "https://api.mistral.ai/v1/chat/completions",
       "parameters": {}
    },
    {
       "model_name": "test",
       "provider_name": "LM Studio",
       "model_id": "qwen3-vl-32b-instruct-uncensored-heretic-i1",
       "provider_id": "LM_Studio",
       "endpoint": "http://127.0.0.1:21234/v1/chat/completions",
       "parameters": {}
    }
]

# Offline (local) vision-language models, executed on-device via OpenVINO GenAI.
# All entries are pre-converted INT4 OpenVINO IR repos on Hugging Face (Apache-2.0 base models).
# "multi_image": whether the model reliably accepts several images in one request;
# when False only the most recent frame is attached to the prompt.
LOCAL_MODELS = [
    {
       # Qwen2-VL is the generation before Qwen2.5-VL/Qwen3-VL above. Kept for the same
       # reason the older Qwen2.5 entry is kept: it is smaller and runs on machines the
       # newer exports are too slow on. The Instruct checkpoint is the one exported here --
       # the plain Qwen2-VL-2B/7B bases are pretrained only and do not follow an
       # instruction like "describe this scene", which is all this app ever asks of them.
       "model_name": "1. 2B, 1.7 GB",
       "provider_name": "Local",
       "model_id": "local-qwen2-vl-2b",
       "provider_id": "local",
       # Not llmware/qwen2-vl-2b-instruct-ov: that export ships no openvino_tokenizer.xml,
       # so VLMPipeline refuses to load it even though the download completes.
       "hf_repo": "helenai/Qwen2-VL-2B-Instruct-ov-int4",
       "size_gb": 1.7,
       "multi_image": False,
       "parameters": {}
    },
    {
       "model_name": "2. 2B, 1.9 GB",
       "provider_name": "Local",
       "model_id": "local-qwen3-vl-2b",
       "provider_id": "local",
       "hf_repo": "llmware/qwen3-vl-2b-ov",
       "size_gb": 1.9,
       "multi_image": True,
       "parameters": {}
    },
    {
       "model_name": "3. 2B uncensored, 2.0 GB",
       "provider_name": "Local",
       "model_id": "local-qwen3-vl-2b-abliterated",
       "provider_id": "local",
       "hf_repo": "EZCon/Huihui-Qwen3-VL-2B-Instruct-abliterated-int4-r0.7-sym-ov",
       "size_gb": 2.0,
       "multi_image": True,
       "parameters": {}
    },
    {
       "model_name": "4. 4B, 3.2 GB",
       "provider_name": "Local",
       "model_id": "local-qwen3-vl-4b",
       "provider_id": "local",
       "hf_repo": "Echo9Zulu/Qwen3-VL-4B-Instruct-int4_asym-ov",
       "size_gb": 3.2,
       "multi_image": True,
       "parameters": {}
    },
    {
       "model_name": "5. 4B uncensored, 3.6 GB",
       "provider_name": "Local",
       "model_id": "local-qwen3-vl-4b-abliterated",
       "provider_id": "local",
       "hf_repo": "EZCon/Huihui-Qwen3-VL-4B-Instruct-abliterated-int4-r0.7-sym-ov",
       "size_gb": 3.6,
       "multi_image": True,
       "parameters": {}
    },
    {
       "model_name": "6. 4B uncensored, 4.5 GB",
       "provider_name": "Local",
       "model_id": "local-qwen3-vl-4b-abliterated-int8",
       "provider_id": "local",
       "hf_repo": "EZCon/Huihui-Qwen3-VL-4B-Instruct-abliterated-int8-r0.7-sym-ov",
       "size_gb": 4.5,
       "multi_image": True,
       "parameters": {}
    },
    {
       "model_name": "7. 7B, 4.8 GB",
       "provider_name": "Local",
       "model_id": "local-qwen2-vl-7b",
       "provider_id": "local",
       "hf_repo": "OpenVINO/Qwen2-VL-7B-Instruct-int4-ov",
       "size_gb": 4.8,
       "multi_image": False,
       "parameters": {}
    },
    {
       "model_name": "8. 7B, 5.2 GB",
       "provider_name": "Local",
       "model_id": "local-qwen2.5-vl-7b",
       "provider_id": "local",
       "hf_repo": "OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov",
       "size_gb": 5.2,
       "multi_image": False,
       "parameters": {}
    },
    {
       "model_name": "9. 8B, 5.5 GB",
       "provider_name": "Local",
       "model_id": "local-qwen3-vl-8b",
       "provider_id": "local",
       "hf_repo": "OpenVINO/Qwen3-VL-8B-Instruct-int4-ov",
       "size_gb": 5.5,
       "multi_image": True,
       "parameters": {}
    },
    {
       # Self-converted (not published anywhere): the original huihui-ai checkpoint is
       # bfloat16, which every other pre-converted int8/int4 export on Hugging Face is built
       # from directly, so it keeps bf16-typed tensors for anything the quantizer leaves
       # unquantized. This one was re-saved from the source checkpoint as float32 first, then
       # quantized to int8, so no bf16 remains anywhere in the model. hf_repo is intentionally
       # absent -- see local_vlm.download_model's guard for models with no re-download source.
       "model_name": "10. 8B uncensored, 8.2 GB",
       "provider_name": "Local",
       "model_id": "local-qwen3-vl-8b-abliterated-int8",
       "provider_id": "local",
       "size_gb": 8.2,
       "multi_image": True,
       "parameters": {}
    }
]

def get_local_models_dir() -> str:
    """
    Where offline models are stored: a "models" folder beside the code itself.

    This used to default to %LOCALAPPDATA% and could be redirected with CINEVOICE_MODELS_DIR,
    which in practice meant a second, slower drive -- and that slow drive was the only reason
    the model cache existed. Keeping the models next to the app removes the indirection, the
    cache, and the copying between them.
    """
    if hasattr(sys, "_MEIPASS"):
        # Beside the exe, not inside _MEIPASS: that folder is the bundle's own payload
        # directory and a onefile build wipes it on exit.
        base = os.path.dirname(os.path.abspath(sys.executable))
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "models")


SEEK_MS = 10_000
MAX_IMAGE_DIM = 448
SETTINGS_FILE = "settings.json"
# Intentionally left as the app's old name: changing this string would make the OS
# credential manager treat it as a different app, orphaning everyone's already-saved
# API keys and forcing them to re-enter every provider's key.
KEYRING_SERVICE_NAME = "AIVideoNarrator"

# Containers that use the ISO base media (MP4/QuickTime) box format: descriptions can be
# embedded directly in the file itself (see mp4_metadata.py). Other formats (.avi, .mkv,
# .wmv) use a different container structure entirely, so they keep using sidecar files.
MP4_METADATA_EXTS = {'.mp4', '.m4v', '.mov'}

# Every format the player can be asked to open. Playback goes through Qt Multimedia's
# FFmpeg backend, which decodes all of these (including still images, which it plays back
# as a single-frame clip), so the same QMediaPlayer/QVideoWidget code path handles both.
VIDEO_EXTS = {'.mp4', '.m4v', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm', '.mpg', '.mpeg',
              '.3gp', '.3g2', '.ts', '.m2ts', '.mts', '.ogv', '.asf', '.vob'}
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff', '.tif', '.ico'}
MEDIA_EXTS = VIDEO_EXTS | IMAGE_EXTS


def _exts_to_patterns(exts: set) -> str:
    return " ".join(sorted(f"*{e}" for e in exts))


# Shared QFileDialog filter string for both the startup picker (main.py) and Ctrl+O (main_window.py).
MEDIA_FILE_DIALOG_FILTER = (
    f"Media Files ({_exts_to_patterns(MEDIA_EXTS)});;"
    f"Video Files ({_exts_to_patterns(VIDEO_EXTS)});;"
    f"Image Files ({_exts_to_patterns(IMAGE_EXTS)});;"
    f"All Files (*.*)"
)

GOOGLE_SPEECH_LANGUAGES = {
    "Afrikaans": "af-ZA", "Arabic": "ar-SA", "Bulgarian": "bg-BG",
    "Catalan": "ca-ES", "Chinese (Mandarin)": "zh-CN", "Chinese (Taiwan)": "zh-TW",
    "Croatian": "hr-HR", "Czech": "cs-CZ", "Danish": "da-DK", "Dutch": "nl-NL",
    "English": "en-US", "English (UK)": "en-GB", "English (Australia)": "en-AU",
    "English (India)": "en-IN", "Filipino": "fil-PH", "Finnish": "fi-FI",
    "French": "fr-FR", "German": "de-DE", "Greek": "el-GR", "Hebrew": "he-IL",
    "Hindi": "hi-IN", "Hungarian": "hu-HU", "Indonesian": "id-ID", "Italian": "it-IT",
    "Japanese": "ja-JP", "Korean": "ko-KR", "Latvian": "lv-LV", "Lithuanian": "lt-LT",
    "Malay": "ms-MY", "Norwegian": "nb-NO", "Persian": "fa-IR", "Polish": "pl-PL",
    "Portuguese (Brazil)": "pt-BR", "Portuguese (Portugal)": "pt-PT", "Romanian": "ro-RO",
    "Russian": "ru-RU", "Serbian": "sr-RS", "Slovak": "sk-SK", "Slovenian": "sl-SI",
    "Spanish": "es-ES", "Spanish (Latin America)": "es-MX", "Swedish": "sv-SE",
    "Thai": "th-TH", "Turkish": "tr-TR", "Ukrainian": "uk-UA", "Vietnamese": "vi-VN"
}