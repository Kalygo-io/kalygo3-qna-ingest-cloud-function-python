"""
Credential decryption — ported from the API microservice's encryption module.

Decrypts the Fernet-encrypted credential blob using CREDENTIALS_ENCRYPTION_KEY
(plus optional comma-separated CREDENTIALS_ENCRYPTION_KEY_OLD for rotation),
fetched from Secret Manager. The decrypted service-account JSON never leaves
function memory and must never be logged.
"""
import json
import logging
from typing import Any, Dict, List, Optional

from cryptography.fernet import Fernet, MultiFernet
from helpers.get_secret import get_secret

logger = logging.getLogger(__name__)

_cached_multi_fernet: Optional[MultiFernet] = None


def _load_keys() -> List[bytes]:
    keys: List[bytes] = []
    primary = get_secret("CREDENTIALS_ENCRYPTION_KEY")
    if not primary:
        raise ValueError("CREDENTIALS_ENCRYPTION_KEY is not available")
    keys.append(primary.encode())

    # Optional rotation keys.
    try:
        old = get_secret("CREDENTIALS_ENCRYPTION_KEY_OLD")
    except Exception:
        old = ""
    if old:
        for k in old.split(","):
            k = k.strip()
            if k:
                keys.append(k.encode())
    return keys


def _get_cipher() -> MultiFernet:
    global _cached_multi_fernet
    if _cached_multi_fernet is None:
        _cached_multi_fernet = MultiFernet([Fernet(k) for k in _load_keys()])
    return _cached_multi_fernet


def decrypt_credential_data(encrypted_data: str) -> Dict[str, Any]:
    """Decrypt an encrypted credential blob into its dict form."""
    if not encrypted_data:
        raise ValueError("Encrypted data cannot be empty")
    cipher = _get_cipher()
    decrypted = cipher.decrypt(encrypted_data.encode())
    return json.loads(decrypted.decode())
