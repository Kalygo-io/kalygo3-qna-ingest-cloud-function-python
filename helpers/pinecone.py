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


def initialize_pinecone() -> Pinecone:
    """
    Initialize Pinecone client.
    
    Returns:
        Pinecone client instance
    """
    api_key = EnvironmentVariables.PINECONE_API_KEY
    
    if not api_key:
        raise ValueError('PINECONE_API_KEY environment variable is required')
    
    logger.info('Initializing Pinecone client with:')
    logger.info(f'- API Key: {api_key[:8]}...')
    
    try:
        pinecone = Pinecone(api_key=api_key)
        logger.info('Pinecone client initialized successfully')
        return pinecone
    except Exception as error:
        logger.error('Error initializing Pinecone client:', exc_info=True)
        raise


def get_pinecone_index(index_name: str = None):
    """
    Get Pinecone index.
    
    Args:
        index_name: Optional index name. If not provided, uses environment variable.
        
    Returns:
        Pinecone index instance
    """
    pinecone = initialize_pinecone()
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


def test_pinecone_connection() -> bool:
    """
    Test Pinecone connection.
    
    Returns:
        True if connection is successful, False otherwise
    """
    try:
        logger.info('Testing Pinecone connection...')
        pinecone = initialize_pinecone()
        
        # Try to list indexes to test connection
        indexes = pinecone.list_indexes()
        logger.info(f'Available indexes: {indexes}')
        
        return True
    except Exception as error:
        logger.error('Pinecone connection test failed:', exc_info=True)
        return False


def upsert_vectors(
    vectors: List[VectorData],
    namespace: str = 'similarity_search'
) -> None:
    """
    Upsert vectors to Pinecone in batches.
    
    Args:
        vectors: List of vector data dictionaries
        namespace: Namespace to upsert vectors to (default: 'similarity_search')
    """
    logger.info(f'Starting upsert of {len(vectors)} vectors to namespace: {namespace}')
    
    # Test connection first
    connection_ok = test_pinecone_connection()
    if not connection_ok:
        raise Exception('Pinecone connection test failed')
    
    index = get_pinecone_index()
    
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

