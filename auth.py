"""API key storage backed by the OS credential manager (via the keyring package)."""
import keyring
from config import KEYRING_SERVICE_NAME, MODELS

def get_unique_providers() -> dict:
    """Returns {provider_id: provider_name} for every online provider in MODELS."""
    providers = {}
    for m in MODELS:
        providers[m["provider_id"]] = m["provider_name"]
    return providers

def save_api_keys(keys_dict: dict) -> tuple:
    """
    Saves each provider's key to the OS keyring; an empty string deletes the stored key.
    "********" is the masked placeholder ApiKeyDialog shows for an existing key, so it's
    skipped here to avoid overwriting the real key with the mask itself.

    Returns (ok, message). A failed keyring write used to be printed to a stdout that a
    windowed build does not have, so the caller went on to announce success and the user
    was left believing a key had been saved that never was.
    """
    failed = []
    for provider_id, key in keys_dict.items():
        if key == "":
            try:
                keyring.delete_password(KEYRING_SERVICE_NAME, f"{provider_id}_key")
            except keyring.errors.PasswordDeleteError:
                pass
            except Exception:
                failed.append(provider_id)
        elif key and key != "********":
            try:
                keyring.set_password(KEYRING_SERVICE_NAME, f"{provider_id}_key", key)
            except Exception:
                failed.append(provider_id)
    if failed:
        names = get_unique_providers()
        listed = ", ".join(names.get(pid, pid) for pid in failed)
        return False, f"Could not save the key for {listed}. Your password manager refused the request."
    return True, "API keys updated."

def load_api_keys() -> dict:
    keys = {}
    for provider_id in get_unique_providers().keys():
        try:
            val = keyring.get_password(KEYRING_SERVICE_NAME, f"{provider_id}_key")
            keys[provider_id] = val if val else ""
        except Exception:
            keys[provider_id] = ""
    return keys