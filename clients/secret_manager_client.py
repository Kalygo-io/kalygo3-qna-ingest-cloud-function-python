"""
Secret Manager Client Factory
"""
from google.cloud import secretmanager


def secret_manager_client_factory() -> secretmanager.SecretManagerServiceClient:
    """
    Create a Secret Manager client.
    For Cloud Functions, use default service account credentials.
    The function will automatically use the service account it's deployed with.
    """
    return secretmanager.SecretManagerServiceClient()

