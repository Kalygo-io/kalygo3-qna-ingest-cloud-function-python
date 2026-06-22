"""
GCS Client Factory
"""
import json
import base64
import logging
from google.cloud import storage
from singletons.environment_variables import EnvironmentVariables

logger = logging.getLogger(__name__)


class AccountGcsCredentialMissing(Exception):
    """Raised when an account has no usable GOOGLE_CLOUD_STORAGE credential."""
    pass


def cloud_storage_client_for_account(account_id: int) -> storage.Client:
    """
    Build a Storage client from the account's own GCS service-account credential.

    Reads the GOOGLE_CLOUD_STORAGE credential row for the account from the DB and
    decrypts it in-memory (the service-account JSON is never logged). Raises
    AccountGcsCredentialMissing if the account has not configured GCS creds.
    """
    # Imported lazily so the module loads even if DB deps are unavailable.
    from db.database import init_database, get_db
    from db.credential_model import Credential
    from db.enums import CredentialType
    from helpers.credential_crypto import decrypt_credential_data

    if not account_id:
        raise AccountGcsCredentialMissing("Missing account_id; cannot resolve GCS credentials")

    init_database()
    db_gen = get_db()
    db = next(db_gen)
    try:
        credential = (
            db.query(Credential)
            .filter(
                Credential.account_id == account_id,
                Credential.credential_type == CredentialType.GOOGLE_CLOUD_STORAGE,
            )
            .order_by(Credential.updated_at.desc())
            .first()
        )
    finally:
        try:
            db.close()
        except Exception:
            pass

    if not credential:
        raise AccountGcsCredentialMissing(
            f"Account {account_id} has no GOOGLE_CLOUD_STORAGE credential configured"
        )

    data = decrypt_credential_data(credential.encrypted_data)
    service_account_json = data.get("service_account_json")
    if not service_account_json:
        raise AccountGcsCredentialMissing(
            f"Account {account_id} GCS credential is missing service_account_json"
        )

    logger.info("Using per-account GCS credentials for account %s", account_id)
    return storage.Client.from_service_account_info(service_account_json)


def cloud_storage_client_factory() -> storage.Client:
    """
    Create a Storage client with proper credentials.
    """
    try:
        kb_ingest_sa = EnvironmentVariables.KB_INGEST_SA
        
        if kb_ingest_sa:
            # Check if KB_INGEST_SA is base64 encoded JSON string
            try:
                key_json = base64.b64decode(kb_ingest_sa).decode('utf-8')
                credentials = json.loads(key_json)
                logger.info('Using service account credentials from KB_INGEST_SA')
                return storage.Client.from_service_account_info(credentials)
            except (base64.binascii.Error, json.JSONDecodeError):
                # If not base64, try as direct JSON string
                try:
                    credentials = json.loads(kb_ingest_sa)
                    logger.info('Using service account credentials from KB_INGEST_SA (direct JSON)')
                    return storage.Client.from_service_account_info(credentials)
                except json.JSONDecodeError:
                    # If not JSON, assume it's a file path
                    import os
                    if os.path.exists(kb_ingest_sa) and kb_ingest_sa.endswith('.json'):
                        logger.info(f'Using service account key file: {kb_ingest_sa}')
                        return storage.Client.from_service_account_json(kb_ingest_sa)
        
        # Fallback to default authentication (Application Default Credentials)
        logger.info('Using default authentication (Application Default Credentials)')
        return storage.Client()
        
    except Exception as error:
        logger.warning(f'Error loading service account credentials, falling back to default auth: {error}')
        return storage.Client()

