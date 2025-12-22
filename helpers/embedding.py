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
        
        # Check for HTTP errors and provide detailed error messages
        if response.status_code == 401:
            error_detail = "Unauthorized - Invalid or expired JWT token"
            logger.error(f'❌ Embedding API returned 401: {error_detail}')
            raise Exception(f'Authentication failed: {error_detail}. Please check your JWT token.')
        elif response.status_code == 403:
            error_detail = "Forbidden - Insufficient permissions"
            logger.error(f'❌ Embedding API returned 403: {error_detail}')
            raise Exception(f'Authorization failed: {error_detail}')
        elif response.status_code == 429:
            error_detail = "Rate limit exceeded"
            logger.error(f'❌ Embedding API returned 429: {error_detail}')
            raise Exception(f'Rate limit exceeded: {error_detail}. Please retry later.')
        elif response.status_code >= 400:
            try:
                error_detail = response.json().get('error', response.text)
            except:
                error_detail = response.text
            logger.error(f'❌ Embedding API returned {response.status_code}: {error_detail}')
            raise Exception(f'Embedding API error ({response.status_code}): {error_detail}')
        
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
        
    except requests.exceptions.Timeout:
        error_msg = 'Embedding API request timed out after 120 seconds'
        logger.error(f'❌ {error_msg}')
        raise Exception(error_msg)
    except requests.exceptions.ConnectionError as e:
        error_msg = f'Failed to connect to embedding API: {str(e)}'
        logger.error(f'❌ {error_msg}')
        raise Exception(error_msg)
    except requests.exceptions.RequestException as e:
        error_msg = f'Embedding API request failed: {str(e)}'
        logger.error(f'❌ {error_msg}')
        raise Exception(error_msg)
    except Exception as error:
        logger.error(f'❌ Error fetching embedding: {error}', exc_info=True)
        # Preserve original error message if it's already descriptive
        if isinstance(error, Exception) and str(error):
            raise
        raise Exception(f'Failed to fetch embedding: {str(error)}')

