"""
Google Cloud Storage helper functions
"""
import logging
from clients.gcs_client_factory import cloud_storage_client_factory

logger = logging.getLogger(__name__)


def download_file_from_gcs(bucket_name: str, file_path: str) -> str:
    """
    Download file from Google Cloud Storage.
    
    Args:
        bucket_name: Name of the GCS bucket
        file_path: Path to the file in the bucket
        
    Returns:
        File content as string (UTF-8)
    """
    try:
        storage_client = cloud_storage_client_factory()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        
        logger.info(f"Downloading file from GCS: gs://{bucket_name}/{file_path}")
        
        # Download the file content
        content = blob.download_as_bytes()
        
        # Convert bytes to string (assuming UTF-8 encoding)
        file_content = content.decode('utf-8')
        
        logger.info(f"Successfully downloaded file: {file_path}, size: {len(content)} bytes")
        
        return file_content
    except Exception as error:
        logger.error(f"Error downloading file from GCS: gs://{bucket_name}/{file_path}", exc_info=True)
        raise Exception(f"Failed to download file from GCS: {str(error)}")


def file_exists_in_gcs(bucket_name: str, file_path: str) -> bool:
    """
    Validate that a file exists in GCS.
    
    Args:
        bucket_name: Name of the GCS bucket
        file_path: Path to the file in the bucket
        
    Returns:
        True if file exists, False otherwise
    """
    try:
        storage_client = cloud_storage_client_factory()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        
        return blob.exists()
    except Exception as error:
        logger.error(f"Error checking if file exists in GCS: gs://{bucket_name}/{file_path}", exc_info=True)
        return False

