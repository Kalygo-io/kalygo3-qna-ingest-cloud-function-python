"""
Enums for database models.
Shared with FastAPI microservice.
"""
from enum import Enum


class OperationType(str, Enum):
    """Enumeration of vector database operation types."""
    INGEST = "INGEST"
    DELETE = "DELETE"
    UPDATE = "UPDATE"


class OperationStatus(str, Enum):
    """Enumeration of vector database operation statuses."""
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"
    PENDING = "PENDING"


class CredentialType(str, Enum):
    """
    Mirror of the API microservice ServiceName enum (PG type credential_type_enum).
    Only used to read/filter the credentials table for per-account GCS creds.
    """
    OPENAI_API_KEY = "OPENAI_API_KEY"
    ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"
    GOOGLE_GEMINI_API_KEY = "GOOGLE_GEMINI_API_KEY"
    PINECONE_API_KEY = "PINECONE_API_KEY"
    ELEVENLABS_API_KEY = "ELEVENLABS_API_KEY"
    SUPABASE = "SUPABASE"
    AWS_SES = "AWS_SES"
    GOOGLE_OAUTH = "GOOGLE_OAUTH"
    GOOGLE_GMAIL_SMTP = "GOOGLE_GMAIL_SMTP"
    GOOGLE_CLOUD_STORAGE = "GOOGLE_CLOUD_STORAGE"

