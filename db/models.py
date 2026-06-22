"""
Database models - shared with FastAPI microservice.
These models match the schema defined in the API microservice.
Migrations are handled in the API microservice, this service only performs CRUD operations.
"""
from sqlalchemy import Column, Integer, String, UUID, JSON, DateTime, func, Text
from sqlalchemy.dialects.postgresql import ENUM as PG_ENUM
from .database import Base
from .enums import OperationType, OperationStatus
import uuid


class VectorDbIngestionLog(Base):
    """
    Logs for VectorDB ingestion operations.
    Tracks what data is being imported into the vector database.
    
    This model matches the schema in the FastAPI microservice.
    """
    __tablename__ = 'vector_db_ingestion_log'
    
    # Primary Key (UUID)
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now(), index=True)
    
    # Operation Details
    # Note: Enum types are created in migration (in API microservice), using create_type=False here
    operation_type = Column(
        PG_ENUM('INGEST', 'DELETE', 'UPDATE', name='operation_type_enum', create_type=False),
        nullable=False,
        index=True
    )
    status = Column(
        PG_ENUM('SUCCESS', 'FAILED', 'PARTIAL', 'PENDING', name='operation_status_enum', create_type=False),
        nullable=False,
        index=True
    )
    
    # User/Account
    # Note: Foreign key constraint exists in DB (created by migrations), but we don't define
    # the Account model here. Since we're only doing CRUD (not creating tables), we don't need
    # to define the ForeignKey in SQLAlchemy - the database will enforce it.
    account_id = Column(Integer, nullable=False, index=True)
    
    # Vector Database Info
    provider = Column(String, nullable=False)  # 'pinecone', 'chroma', etc.
    index_name = Column(String, nullable=False, index=True)
    namespace = Column(String, nullable=True, index=True)
    
    # File Information
    filenames = Column(JSON, nullable=True)  # Array of filenames
    comment = Column(Text, nullable=True)

    # Pointer back to the original source document in Google Cloud Storage.
    gcs_bucket = Column(String, nullable=True)
    gcs_file_path = Column(String, nullable=True)

    # Vector Counts
    vectors_added = Column(Integer, default=0)
    vectors_deleted = Column(Integer, default=0)
    vectors_failed = Column(Integer, default=0)
    
    # Error Handling
    error_message = Column(Text, nullable=True)
    error_code = Column(String, nullable=True)
    
    # Batch Grouping
    batch_number = Column(String, nullable=True, index=True)  # UUID for grouping related operations
    
    # Note: Account relationship would require importing Account model
    # For this microservice, we only need to write logs, not read relationships
    
    def __repr__(self):
        return f'<VectorDbIngestionLog {self.id} - {self.operation_type} - {self.status}>'

