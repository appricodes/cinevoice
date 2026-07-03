"""
Downloads pre-converted OpenVINO vision-language models from Hugging Face and runs
them on-device via openvino-genai, preferring an Intel GPU with CPU fallback.
openvino/openvino_genai/cv2/numpy are imported lazily inside functions so app
startup stays fast for users who never touch the offline-model feature.
"""
import os
import re
import gc
import shutil
import base64
import threading

from config import LOCAL_MODELS, get_local_models_dir

DOWNLOAD_MARKER = ".download_complete"

_engine_lock = threading.Lock()
_pipeline = None
_pipeline_dir = None
_device_used = None


def get_model_dir(model_dict: dict) -> str:
    return os.path.join(get_local_models_dir(), model_dict["model_id"])


def is_model_downloaded(model_dict: dict) -> bool:
    model_dir = get_model_dir(model_dict)
    return (os.path.isfile(os.path.join(model_dir, DOWNLOAD_MARKER))
            and os.path.isfile(os.path.join(model_dir, "openvino_language_model.xml")))


def get_downloaded_local_models() -> list:
    return [m for m in LOCAL_MODELS if is_model_downloaded(m)]


def find_local_model(model_id: str) -> dict:
    for m in LOCAL_MODELS:
        if m["model_id"] == model_id:
            return m
    return None


def delete_model(model_dict: dict) -> tuple:
    """
    Permanently removes a downloaded model's directory from disk. Unloads it
    from memory first if it is the currently active pipeline, since OpenVINO
    can hold open file handles on the model's weights while loaded, which
    would otherwise make deletion fail on Windows.
    Returns (success: bool, message: str).
    """
    model_dir = get_model_dir(model_dict)
    if _pipeline_dir == model_dir:
        unload_pipeline()
    if not os.path.isdir(model_dir):
        return True, f"{model_dict['model_name']} is not on disk."
    try:
        shutil.rmtree(model_dir)
    except Exception as e:
        return False, f"Could not delete {model_dict['model_name']}: {e}"
    return True, f"{model_dict['model_name']} has been deleted from your computer."


def _list_repo_files(repo: str) -> list:
    """Returns [(path, size), ...] for every file in a Hugging Face repository."""
    import requests
    url = f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return [(item["path"], item.get("size", 0) or 0)
            for item in resp.json() if item.get("type") == "file"]


def download_model(model_dict: dict, progress_cb=None, cancel_event=None) -> tuple:
    """
    Downloads every file of the model repository into the local models directory
    with streamed chunks, so progress is smooth and cancellation reacts within
    a second. Interrupted files resume from where they stopped via HTTP ranges.
    Reports size-weighted percentage through progress_cb(int).
    Returns (success: bool, message: str).
    """
    import requests
    from urllib.parse import quote

    repo = model_dict["hf_repo"]
    target_dir = get_model_dir(model_dict)
    os.makedirs(target_dir, exist_ok=True)

    try:
        files = _list_repo_files(repo)
    except Exception as e:
        return False, f"Could not reach Hugging Face: {e}"
    if not files:
        return False, "The model repository appears to be empty."

    total_bytes = sum(size for _, size in files) or 1
    done_bytes = 0

    def report():
        if progress_cb:
            progress_cb(min(100, int(done_bytes * 100 / total_bytes)))

    session = requests.Session()
    try:
        for path, size in files:
            if cancel_event is not None and cancel_event.is_set():
                return False, "Download cancelled. It will resume from this point next time."

            dest = os.path.join(target_dir, *path.split("/"))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if os.path.isfile(dest) and (size == 0 or os.path.getsize(dest) == size):
                done_bytes += size
                report()
                continue

            part = dest + ".part"
            resume_from = os.path.getsize(part) if os.path.isfile(part) else 0
            headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
            url = f"https://huggingface.co/{repo}/resolve/main/{quote(path)}"

            with session.get(url, headers=headers, stream=True, timeout=60) as resp:
                if resume_from and resp.status_code == 200:
                    resume_from = 0  # server ignored the range request; restart the file
                resp.raise_for_status()
                done_bytes += resume_from
                with open(part, "ab" if resume_from else "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if cancel_event is not None and cancel_event.is_set():
                            return False, "Download cancelled. It will resume from this point next time."
                        f.write(chunk)
                        done_bytes += len(chunk)
                        report()

            if size and os.path.getsize(part) != size:
                return False, f"Download of {path} was incomplete. Please try again."
            os.replace(part, dest)
    except Exception as e:
        return False, f"Download failed: {e}. It will resume next time."

    try:
        with open(os.path.join(target_dir, DOWNLOAD_MARKER), "w") as f:
            f.write("ok")
    except Exception:
        pass
    return True, "Download complete."


def detect_device() -> str:
    """
    Prefers a GPU device when OpenVINO exposes a usable one, otherwise CPU.
    OpenVINO's GPU plugin only supports Intel graphics; it enumerates other
    vendors' cards too but fails at inference time, so those are skipped.
    """
    try:
        import openvino as ov
        core = ov.Core()
        for d in core.available_devices:
            if not d.startswith("GPU"):
                continue
            try:
                name = core.get_property(d, "FULL_DEVICE_NAME")
            except Exception:
                continue
            if "intel" in str(name).lower():
                return d
    except Exception:
        pass
    return "CPU"


def unload_pipeline():
    global _pipeline, _pipeline_dir, _device_used
    _pipeline = None
    _pipeline_dir = None
    _device_used = None
    gc.collect()


def _get_pipeline(model_dir: str, status_cb=None, force_device: str = None):
    """Loads (or reuses) the VLM pipeline for the given model directory."""
    global _pipeline, _pipeline_dir, _device_used
    import openvino_genai as ov_genai

    if _pipeline is not None and _pipeline_dir == model_dir and (force_device is None or _device_used == force_device):
        return _pipeline

    unload_pipeline()
    device = force_device or detect_device()
    if status_cb:
        status_cb(f"Loading offline model into memory on {device}. This can take a minute...")
    try:
        _pipeline = ov_genai.VLMPipeline(model_dir, device)
    except Exception:
        if device == "CPU":
            raise
        if status_cb:
            status_cb("GPU load failed. Falling back to CPU...")
        device = "CPU"
        _pipeline = ov_genai.VLMPipeline(model_dir, device)
    _pipeline_dir = model_dir
    _device_used = device
    return _pipeline


def _b64_to_tensor(frame_b64: str):
    """Decodes a base64 JPEG into an RGB uint8 ov.Tensor as expected by VLMPipeline."""
    import cv2
    import numpy as np
    import openvino as ov
    data = np.frombuffer(base64.b64decode(frame_b64), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode captured frame.")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return ov.Tensor(rgb)


def _fold_history(history: list, current_text: str) -> str:
    """
    Flattens prior conversation turns into the prompt since the local pipeline
    is used statelessly. Skips a trailing duplicate of the current user text
    (the caller appends the current prompt to history before sending).
    """
    if not history:
        return current_text
    turns = list(history)
    if turns and turns[-1].get("role") == "user" and turns[-1].get("content") == current_text:
        turns = turns[:-1]
    if not turns:
        return current_text
    lines = ["Previous conversation for context:"]
    for t in turns:
        role = "User" if t.get("role") == "user" else "Assistant"
        content = t.get("content", "")
        if isinstance(content, str) and content:
            lines.append(f"{role}: {content}")
    lines.append("")
    lines.append(current_text)
    return "\n".join(lines)


def _enforce_language(prompt: str, language: str) -> str:
    """
    Appends an explicit, unambiguous language directive at the very end of the
    prompt (closest to generation). Small quantized VLMs follow instructions
    placed early in a long system prompt inconsistently for some languages
    (verified: local models can silently fall back to English even when the
    system prompt already requests a different language). Recency in the
    prompt strongly biases the next tokens, so repeating the instruction here
    is far more reliable than relying on the system prompt alone.
    """
    if not language or language.strip().lower() == "english":
        return prompt
    return (f"{prompt}\n\n"
            f"IMPORTANT: Write your entire response only in {language}. "
            f"Do not use English or any other language.")


def generate(model_dict: dict, frames_b64: list, system_prompt: str, user_text: str,
             history: list = None, max_new_tokens: int = 1024, status_cb=None, language: str = None) -> str:
    """
    Runs one full multimodal generation on the local model and returns plain text.
    Serialized by a lock: only one local inference can run at a time.
    """
    model_dir = get_model_dir(model_dict)
    if not is_model_downloaded(model_dict):
        raise RuntimeError("This offline model is not downloaded. Open Settings with Control plus S to download it.")

    frames = list(frames_b64 or [])
    if not model_dict.get("multi_image", False) and len(frames) > 1:
        frames = frames[-1:]

    prompt = _fold_history(history, user_text)
    prompt = _enforce_language(prompt, language)

    with _engine_lock:
        pipe = _get_pipeline(model_dir, status_cb)
        tensors = [_b64_to_tensor(b) for b in frames]
        if status_cb:
            status_cb(f"Offline model is generating on {_device_used}. This may be slow...")

        def _run(images):
            pipe.start_chat(system_prompt)
            try:
                return pipe.generate(prompt, images=images,
                                     max_new_tokens=max_new_tokens, do_sample=False)
            finally:
                pipe.finish_chat()

        try:
            try:
                result = _run(tensors)
            except Exception:
                if len(tensors) > 1:
                    # Some converted models only accept a single image; retry with the last frame.
                    result = _run(tensors[-1:])
                else:
                    raise
        except Exception:
            if _device_used and _device_used.startswith("GPU"):
                # GPU generation can fail on cards with little memory; retry fully on CPU.
                if status_cb:
                    status_cb("GPU generation failed. Retrying on CPU...")
                pipe = _get_pipeline(model_dir, status_cb, force_device="CPU")
                result = _run(tensors if model_dict.get("multi_image", False) else tensors[-1:])
            else:
                raise

    text = result.texts[0] if hasattr(result, "texts") and result.texts else str(result)
    # Strip reasoning blocks that thinking-variant checkpoints may emit.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()
