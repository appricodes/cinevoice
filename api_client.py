"""
Sends vision/chat requests to any OpenAI-compatible provider (Grok, Gemini, OpenAI,
Mistral), or routes to the local OpenVINO model when the selected model is offline.
"""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class MultiClient:
    """One client for every online provider, since they all share the OpenAI chat/completions shape."""
    def __init__(self, api_keys_dict: dict):
        self.keys = api_keys_dict
        self.status_cb = None  # optional callable(str) used by local models to report progress
        # {"usage": <the response's usage dict>, "provider_id": str, "model_name": str} for
        # whichever request most recently got a real reply, read back by Ctrl+P. None until
        # the first call completes, or after a local-model call (those report no token usage).
        self.last_call_info = None
        self.session = requests.Session()
        retries = Retry(
            total=3, 
            backoff_factor=0.6,
            status_forcelist=(408, 429, 500, 502, 503, 504),
            allowed_methods=frozenset(['POST'])
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
       
    def send(self, model_dict: dict, frames_b64: list, system_prompt: str, user_text_blocks: list, expect_json: bool = False, history: list = None, language: str = None, stream_cb=None) -> dict:
        provider = model_dict.get("provider_id")
        model = model_dict.get("model_id")
        url = model_dict.get("endpoint")

        if provider == "local":
            self.last_call_info = None
            return self._send_local(model_dict, frames_b64, system_prompt, user_text_blocks, history, language, stream_cb)


        api_key = self.keys.get(provider)
        if not api_key:
            return {"error": f"API Key for {model_dict.get('provider_name')} is missing. Press Ctrl+S to update your keys."}
           
        text = "\n".join([t for t in user_text_blocks if t])
        content = [{"type": "text", "text": text}]
        
        if frames_b64:
            for b64 in frames_b64:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": content})
        
        payload = {"model": model, "messages": messages}
        
        parameters = model_dict.get("parameters", {})
        payload.update(parameters)
            
        if expect_json:
            payload["response_format"] = {"type": "json_object"}
            
        headers = {
            "Authorization": f"Bearer {api_key}", 
            "Content-Type": "application/json"
        }
        
        resp = None
        try:
            resp = self.session.post(url, headers=headers, json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            self.last_call_info = {
                "usage": data.get("usage") or {},
                "provider_id": provider,
                "model_name": model_dict.get("model_name"),
            }
            return data
        except requests.exceptions.HTTPError as e:
            # resp is normally set by the line above, but requests can raise HTTPError from
            # the post itself; reading resp.status_code then would be a NameError, replacing
            # the real error with a crash.
            if resp is None:
                return {"error": f"API Error ({provider}): {e}"}
            return {"error": f"API Error ({provider}): Status {resp.status_code}. {resp.text}"}
        except Exception as e:
            return {"error": f"Network Error: {str(e)}"}

    def _send_local(self, model_dict: dict, frames_b64: list, system_prompt: str, user_text_blocks: list, history: list = None, language: str = None, stream_cb=None) -> dict:
        # Wraps the result in the same {"choices": [...]} shape as the online providers
        # above, so callers never need to know whether a model ran locally or remotely.
        try:
            import local_vlm
            user_text = "\n".join([t for t in user_text_blocks if t])
            text = local_vlm.generate(
                model_dict=model_dict,
                frames_b64=frames_b64,
                system_prompt=system_prompt,
                user_text=user_text,
                history=history,
                status_cb=self.status_cb,
                language=language,
                chunk_cb=stream_cb
            )
            return {"choices": [{"message": {"content": text}}]}
        except Exception as e:
            return {"error": f"Offline model error: {e}"}