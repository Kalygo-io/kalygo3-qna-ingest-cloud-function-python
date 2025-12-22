"""
Embedding helper functions
"""
import logging
import requests
from singletons.environment_variables import EnvironmentVariables

logger = logging.getLogger(__name__)


def fetch_embedding(jwt: str, text: str) -> list[float]:
    """
    Fetch embedding from the embedding API service.
    
    Args:
        jwt: JWT token for authentication
        text: Text to embed
        
    Returns:
        The embedding vector as a list of floats
    """
    try:
        print('--- --- --- --- ---')
        print(f'Fetching embedding for text: {text}')
        print('--- --- --- --- ---')
        print(f'Embedding API URL: {EnvironmentVariables.EMBEDDINGS_API_URL}')
        print('--- --- --- --- ---')
        print(f'JWT: {jwt}')
        print('--- --- --- --- ---')

        embeddings_api_url = f'{EnvironmentVariables.EMBEDDINGS_API_URL}/huggingface/embedding'
        
        response = requests.post(
            embeddings_api_url,
            json={'input': text},
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {jwt}'
            },
            timeout=120  # 120 second timeout
        )
        
        response.raise_for_status()
        
        logger.info(f'Embedding API response structure: {list(response.json().keys())}')
        logger.info(f'Response data type: {type(response.json())}')
        
        response_data = response.json()
        
        # Handle different possible response formats
        embedding_array = None
        
        if response_data and 'embedding' in response_data:
            embedding_array = response_data['embedding']
        elif response_data and isinstance(response_data, list):
            embedding_array = response_data
        elif response_data and 'data' in response_data and isinstance(response_data['data'], list):
            embedding_array = response_data['data']
        else:
            logger.error(f'Unexpected response format: {response_data}')
            raise ValueError('Invalid response format from embedding API')
        
        # Ensure all values are numbers and flatten if necessary
        flattened_embedding: list[float] = []
        
        def flatten_array(arr):
            """Recursively flatten nested arrays."""
            for item in arr:
                if isinstance(item, list):
                    flatten_array(item)
                else:
                    try:
                        num = float(item)
                        if not isinstance(num, (int, float)) or (isinstance(num, float) and (num != num)):  # Check for NaN
                            logger.error(f'Invalid embedding value: {item}, type: {type(item)}')
                            raise ValueError(f'Invalid embedding value: {item}')
                        flattened_embedding.append(num)
                    except (ValueError, TypeError) as e:
                        logger.error(f'Invalid embedding value: {item}, type: {type(item)}')
                        raise ValueError(f'Invalid embedding value: {item}') from e
        
        flatten_array(embedding_array)
        
        logger.info(f'Generated embedding with {len(flattened_embedding)} dimensions')
        logger.info(f'First few values: {flattened_embedding[:5]}')
        
        return flattened_embedding
        
    except Exception as error:
        logger.error(f'Error fetching embedding: {error}', exc_info=True)
        raise Exception(f'Failed to fetch embedding: {str(error)}')

