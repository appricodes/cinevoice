"""API key storage backed by the OS credential manager (via the keyring package)."""
import keyring
from config import KEYRING_SERVICE_NAME, MODELS

def get_unique_providers() -> dict:
    """Returns {provider_id: provider_name} for every online provider in MODELS."""
    providers = {}
    for m in MODELS:
        providers[m["provider_id"]] = m["provider_name"]
    return providers

def save_api_keys(keys_dict: dict):
    """
    Saves each provider's key to the OS keyring; an empty string deletes the stored key.
    "********" is the masked placeholder ApiKeyDialog shows for an existing key, so it's
    skipped here to avoid overwriting the real key with the mask itself.
    """
    for provider_id, key in keys_dict.items():
        if key == "":
            try:
                keyring.delete_password(KEYRING_SERVICE_NAME, f"{provider_id}_key")
            except keyring.errors.PasswordDeleteError:
                pass
        elif key and key != "********":
            try:
                keyring.set_password(KEYRING_SERVICE_NAME, f"{provider_id}_key", key)
            except Exception as e:
                print(f"Error saving {provider_id} key: {e}")

def load_api_keys() -> dict:
    keys = {}
    for provider_id in get_unique_providers().keys():
        try:
            val = keyring.get_password(KEYRING_SERVICE_NAME, f"{provider_id}_key")
            keys[provider_id] = val if val else ""
        except Exception:
            keys[provider_id] = ""
    return keys