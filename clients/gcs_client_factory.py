"""
GCS Client Factory
"""
import json
import base64
import logging
from google.cloud import storage
from singletons.environment_variables import EnvironmentVariables

logger = logging.getLogger(__name__)


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

