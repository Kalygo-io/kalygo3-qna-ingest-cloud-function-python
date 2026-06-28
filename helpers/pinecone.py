"""
Pinecone helper functions
"""
import logging
from typing import Dict, Any, List
from pinecone import Pinecone
from singletons.environment_variables import EnvironmentVariables

logger = logging.getLogger(__name__)


# Type definitions
VectorData = Dict[str, Any]  # {'id': str, 'values': List[float], 'metadata': Dict[str, Any]}
ProcessingResult = Dict[str, Any]


def pinecone_api_key_for_account(account_id: int) -> str:
    """
    Resolve an account's own Pinecone API key from the credentials table.

    Reads the PINECONE_API_KEY credential row for the account and decrypts it
    in-memory (the key is never logged). This makes ingestion use the SAME
    per-account Pinecone project the API microservice reads/manages with, so the
    chosen index actually receives the data. Raises ValueError if the account has
    no Pinecone credential configured.
    """
    # Imported lazily so the module loads even if DB deps are unavailable
    # (mirrors clients/gcs_client_factory.py).
    from db.database import init_database, get_db
    from db.credential_model import Credential
    from db.enums import CredentialType
    from helpers.credential_crypto import decrypt_credential_data

    if not account_id:
        raise ValueError("Missing account_id; cannot resolve Pinecone credentials")

    init_database()
    db_gen = get_db()
    db = next(db_gen)
    try:
        credential = (
            db.query(Credential)
            .filter(
                Credential.account_id == account_id,
                Credential.credential_type == CredentialType.PINECONE_API_KEY,
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
        raise ValueError(f"Account {account_id} has no PINECONE_API_KEY credential configured")

    data = decrypt_credential_data(credential.encrypted_data)
    api_key = data.get("api_key")
    if not api_key:
        raise ValueError(f"Account {account_id} Pinecone credential is missing api_key")

    logger.info("Using per-account Pinecone credentials for account %s", account_id)
    return api_key


def initialize_pinecone(api_key: str = None) -> Pinecone:
    """
    Initialize Pinecone client.

    Args:
        api_key: Account's Pinecone API key. When omitted, falls back to the
            globally configured PINECONE_API_KEY (legacy/default).

    Returns:
        Pinecone client instance
    """
    api_key = api_key or EnvironmentVariables.PINECONE_API_KEY

    if not api_key:
        raise ValueError('A Pinecone API key is required')

    # Never log the key (or any prefix of it).
    logger.info('Initializing Pinecone client')

    try:
        pinecone = Pinecone(api_key=api_key)
        logger.info('Pinecone client initialized successfully')
        return pinecone
    except Exception as error:
        logger.error('Error initializing Pinecone client:', exc_info=True)
        raise


def get_pinecone_index(index_name: str = None, api_key: str = None):
    """
    Get Pinecone index.

    Args:
        index_name: Optional index name. If not provided, uses environment variable.
        api_key: Optional account Pinecone API key (see initialize_pinecone).

    Returns:
        Pinecone index instance
    """
    pinecone = initialize_pinecone(api_key)
    index_name = index_name or EnvironmentVariables.PINECONE_ALL_MINILM_L6_V2_INDEX
    
    if not index_name:
        raise ValueError('PINECONE_ALL_MINILM_L6_V2_INDEX environment variable is required')
    
    logger.info(f'Getting Pinecone index: {index_name}')
    
    try:
        index = pinecone.Index(index_name)
        logger.info('Pinecone index retrieved successfully')
        return index
    except Exception as error:
        logger.error('Error getting Pinecone index:', exc_info=True)
        raise


def test_pinecone_connection(api_key: str = None) -> bool:
    """
    Test Pinecone connection.

    Returns:
        True if connection is successful, False otherwise
    """
    try:
        logger.info('Testing Pinecone connection...')
        pinecone = initialize_pinecone(api_key)
        
        # Try to list indexes to test connection
        indexes = pinecone.list_indexes()
        logger.info(f'Available indexes: {indexes}')
        
        return True
    except Exception as error:
        logger.error('Pinecone connection test failed:', exc_info=True)
        return False


def upsert_vectors(
    vectors: List[VectorData],
    namespace: str = 'similarity_search',
    index_name: str = None,
    api_key: str = None
) -> None:
    """
    Upsert vectors to Pinecone in batches.

    Args:
        vectors: List of vector data dictionaries
        namespace: Namespace to upsert vectors to (default: 'similarity_search')
        index_name: Target Pinecone index. When omitted, falls back to the
            configured default index (PINECONE_ALL_MINILM_L6_V2_INDEX).
        api_key: Account's Pinecone API key (see initialize_pinecone). When
            omitted, falls back to the globally configured key.
    """
    logger.info(f'Starting upsert of {len(vectors)} vectors to index: {index_name or "<default>"}, namespace: {namespace}')

    # Test connection first
    connection_ok = test_pinecone_connection(api_key)
    if not connection_ok:
        raise Exception('Pinecone connection test failed')

    index = get_pinecone_index(index_name, api_key)
    
    try:
        logger.info('📊 Vectors to be inserted:')
        logger.info(f'   Count: {len(vectors)}')
        if vectors:
            logger.info(f'   First vector ID: {vectors[0].get("id", "")[:12]}...')
            logger.info(f'   First vector dimensions: {len(vectors[0].get("values", []))}')
            logger.info(f'   Sample values from first vector: {vectors[0].get("values", [])[:3]}')
        logger.info(f'   Namespace: {namespace}')
    except Exception as diagnostic_error:
        logger.warning(f'⚠️  Diagnostic check failed: {str(diagnostic_error)}')
    
    # Pinecone recommends batch sizes of 100 or less
    batch_size = 100
    
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(vectors) + batch_size - 1) // batch_size
        
        try:
            logger.info(f'Upserting batch {batch_num} with {len(batch)} vectors')
            index.upsert(vectors=batch, namespace=namespace)
            logger.info(f'Successfully upserted batch {batch_num} of {total_batches}')
        except Exception as error:
            logger.error(f'Error upserting batch {batch_num}:', exc_info=True)
            raise

