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
        self.session = requests.Session()
        retries = Retry(
            total=3, 
            backoff_factor=0.6,
            status_forcelist=(408, 429, 500, 502, 503, 504),
            allowed_methods=frozenset(['POST'])
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
       
    def send(self, model_dict: dict, frames_b64: list, system_prompt: str, user_text_blocks: list, expect_json: bool = False, history: list = None, language: str = None) -> dict:
        provider = model_dict.get("provider_id")
        model = model_dict.get("model_id")
        url = model_dict.get("endpoint")

        if provider == "local":
            return self._send_local(model_dict, frames_b64, system_prompt, user_text_blocks, history, language)


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
        
        try:
            resp = self.session.post(url, headers=headers, json=payload, timeout=300)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            return {"error": f"API Error ({provider}): Status {resp.status_code}. {resp.text}"}
        except Exception as e:
            return {"error": f"Network Error: {str(e)}"}

    def _send_local(self, model_dict: dict, frames_b64: list, system_prompt: str, user_text_blocks: list, history: list = None, language: str = None) -> dict:
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
                language=language
            )
            return {"choices": [{"message": {"content": text}}]}
        except Exception as e:
            return {"error": f"Offline model error: {e}"}