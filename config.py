"""Static configuration: AI model catalog, tunable constants, and language tables."""
import os

MODELS = [
    {
       "model_name": "Grok 4.3",
       "provider_name": "xAI",
       "model_id": "grok-4.3",
       "provider_id": "grok",
       "endpoint": "https://api.x.ai/v1/chat/completions",
       "parameters": {
          "reasoning_effort": "none"
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
       "model_name": "Mistral ministral 3b",
       "provider_name": "Mistral",
       "model_id": "ministral-3b-latest",
       "provider_id": "mistral",
       "endpoint": "https://api.mistral.ai/v1/chat/completions",
       "parameters": {
          "reasoning_effort": "none"
       }
    },
    {
       "model_name": "Mistral ministral 14b",
       "provider_name": "Mistral",
       "model_id": "ministral-14b-latest",
       "provider_id": "mistral",
       "endpoint": "https://api.mistral.ai/v1/chat/completions",
       "parameters": {
          "reasoning_effort": "none"
       }
    },
    {
       "model_name": "Mistral Small",
       "provider_name": "Mistral",
       "model_id": "mistral-small-latest",
       "provider_id": "mistral",
       "endpoint": "https://api.mistral.ai/v1/chat/completions",
       "parameters": {
          "reasoning_effort": "none"
       }
    }
]

# Offline (local) vision-language models, executed on-device via OpenVINO GenAI.
# All entries are pre-converted INT4 OpenVINO IR repos on Hugging Face (Apache-2.0 base models).
# "multi_image": whether the model reliably accepts several images in one request;
# when False only the most recent frame is attached to the prompt.
LOCAL_MODELS = [
    {
       "model_name": "Offline Qwen3-VL 2B (1.9 GB, fastest)",
       "provider_name": "Local",
       "model_id": "local-qwen3-vl-2b",
       "provider_id": "local",
       "hf_repo": "llmware/qwen3-vl-2b-ov",
       "size_gb": 1.9,
       "multi_image": True,
       "parameters": {}
    },
    {
       "model_name": "Offline Qwen3-VL 4B (3.2 GB, balanced)",
       "provider_name": "Local",
       "model_id": "local-qwen3-vl-4b",
       "provider_id": "local",
       "hf_repo": "Echo9Zulu/Qwen3-VL-4B-Instruct-int4_asym-ov",
       "size_gb": 3.2,
       "multi_image": True,
       "parameters": {}
    },
    {
       "model_name": "Offline Qwen2.5-VL 7B (5.2 GB, high quality)",
       "provider_name": "Local",
       "model_id": "local-qwen2.5-vl-7b",
       "provider_id": "local",
       "hf_repo": "OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov",
       "size_gb": 5.2,
       "multi_image": False,
       "parameters": {}
    },
    {
       "model_name": "Offline Qwen3-VL 8B (5.5 GB, best quality)",
       "provider_name": "Local",
       "model_id": "local-qwen3-vl-8b",
       "provider_id": "local",
       "hf_repo": "OpenVINO/Qwen3-VL-8B-Instruct-int4-ov",
       "size_gb": 5.5,
       "multi_image": True,
       "parameters": {}
    },
    {
       "model_name": "Offline Qwen3-VL 2B Uncensored (2.0 GB)",
       "provider_name": "Local",
       "model_id": "local-qwen3-vl-2b-abliterated",
       "provider_id": "local",
       "hf_repo": "EZCon/Huihui-Qwen3-VL-2B-Instruct-abliterated-int4-r0.7-sym-ov",
       "size_gb": 2.0,
       "multi_image": True,
       "parameters": {}
    },
    {
       "model_name": "Offline Qwen3-VL 4B Uncensored (3.6 GB)",
       "provider_name": "Local",
       "model_id": "local-qwen3-vl-4b-abliterated",
       "provider_id": "local",
       "hf_repo": "EZCon/Huihui-Qwen3-VL-4B-Instruct-abliterated-int4-r0.7-sym-ov",
       "size_gb": 3.6,
       "multi_image": True,
       "parameters": {}
    },
    {
       "model_name": "Offline Qwen3-VL 4B Uncensored, higher quality (4.5 GB, int8)",
       "provider_name": "Local",
       "model_id": "local-qwen3-vl-4b-abliterated-int8",
       "provider_id": "local",
       "hf_repo": "EZCon/Huihui-Qwen3-VL-4B-Instruct-abliterated-int8-r0.7-sym-ov",
       "size_gb": 4.5,
       "multi_image": True,
       "parameters": {}
    }
]

def get_local_models_dir() -> str:
    """Resolves the per-user directory where offline models are stored (outside the exe)."""
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    app_dir = os.path.join(base, "Cinevoice")
    old_app_dir = os.path.join(base, "AIVideoPlayer")
    # One-time migration from the app's former name, so multi-gigabyte models
    # already downloaded by existing users aren't orphaned under the old folder.
    # Moves each model subfolder individually rather than renaming the old parent
    # folder as a whole: some process (an editor, terminal, or antivirus scan) can
    # hold the parent folder itself locked as a working directory even though
    # nothing inside it is locked, which makes a single top-level rename fail.
    if not os.path.isdir(app_dir) and os.path.isdir(old_app_dir):
        import shutil
        old_models_dir = os.path.join(old_app_dir, "models")
        new_models_dir = os.path.join(app_dir, "models")
        if os.path.isdir(old_models_dir):
            os.makedirs(new_models_dir, exist_ok=True)
            for name in os.listdir(old_models_dir):
                try:
                    shutil.move(os.path.join(old_models_dir, name), os.path.join(new_models_dir, name))
                except Exception:
                    pass
        try:
            os.rmdir(old_models_dir)
            os.rmdir(old_app_dir)
        except Exception:
            pass
    return os.path.join(app_dir, "models")

SEEK_MS = 10_000
MAX_IMAGE_DIM = 448
SETTINGS_FILE = "settings.json"
# Intentionally left as the app's old name: changing this string would make the OS
# credential manager treat it as a different app, orphaning everyone's already-saved
# API keys and forcing them to re-enter every provider's key.
KEYRING_SERVICE_NAME = "AIVideoNarrator"

SCENE_KMEAN_IMAGE_SIZE = (64, 64)
SCENE_FRAME_INTERVAL = 3.0
SCENE_KMEAN_CLUSTERS = 3

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