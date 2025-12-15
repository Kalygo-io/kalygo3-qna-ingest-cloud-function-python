"""
Secret Manager helper functions
"""
import logging
from typing import Dict
from clients.secret_manager_client import secret_manager_client_factory

logger = logging.getLogger(__name__)

# Cache for secrets
_secrets_cache: Dict[str, str] = {}


def get_secret(secret_name: str) -> str:
    """
    Get a secret from Google Secret Manager.
    
    Args:
        secret_name: Name of the secret
        
    Returns:
        Secret value as string
    """
    # Check cache first
    if secret_name in _secrets_cache:
        return _secrets_cache[secret_name]
    
    try:
        secret_manager = secret_manager_client_factory()
        project_id = "830723611668"  # Hardcoded project ID from Node.js version
        
        # Access the secret version
        name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
        response = secret_manager.access_secret_version(request={"name": name})
        
        secret = response.payload.data.decode("UTF-8")
        
        # Cache the secret for subsequent calls
        _secrets_cache[secret_name] = secret
        
        return secret
    except Exception as error:
        logger.error(f"Error retrieving secret {secret_name}: {error}")
        raise

