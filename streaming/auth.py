"""Kalshi WebSocket handshake authentication with environment-only secrets."""

import base64
import os
import time
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


WS_SIGN_PATH = "/trade-api/ws/v2"


class CredentialError(RuntimeError):
    pass


def load_dotenv(path=".env"):
    """Small non-printing .env loader; existing environment always wins."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def load_credentials(dotenv_path=".env"):
    load_dotenv(dotenv_path)
    key_id = os.environ.get("KALSHI_KEY_ID")
    key_file = os.environ.get("KALSHI_KEY_FILE")
    if not key_id or not key_file:
        raise CredentialError(
            "WebSocket mode requires KALSHI_KEY_ID and KALSHI_KEY_FILE in the "
            "environment or ignored .env; REST diagnostics remain available")
    path = Path(key_file).expanduser()
    if not path.is_file():
        raise CredentialError(f"KALSHI_KEY_FILE does not exist: {path}")
    return key_id, path


def build_auth_headers(key_id, private_key_pem, timestamp_ms=None):
    timestamp = str(timestamp_ms if timestamp_ms is not None else int(time.time() * 1000))
    try:
        private_key = serialization.load_pem_private_key(private_key_pem, password=None)
    except (ValueError, TypeError) as exc:
        raise CredentialError("KALSHI_KEY_FILE is not a valid unencrypted PEM key") from exc
    message = f"{timestamp}GET{WS_SIGN_PATH}".encode()
    signature = private_key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256())
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("ascii"),
    }


def auth_headers_from_environment(dotenv_path=".env"):
    key_id, key_path = load_credentials(dotenv_path)
    return build_auth_headers(key_id, key_path.read_bytes())
