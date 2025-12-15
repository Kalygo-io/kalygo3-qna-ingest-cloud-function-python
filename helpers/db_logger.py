"""
Database logging helper for VectorDB ingestion operations.
Provides a simple interface for logging ingestion operations to the database.
"""
import logging
import uuid
from typing import Optional, List
from db.database import get_db, init_database
from db.models import VectorDbIngestionLog
from db.enums import OperationType, OperationStatus

logger = logging.getLogger(__name__)


class VectorDbLogger:
    """
    Helper class for logging VectorDB ingestion operations to the database.
    """
    
    def __init__(self, account_id: int, provider: str = "pinecone", batch_number: Optional[str] = None):
        """
        Initialize logger for a batch of operations.
        
        Args:
            account_id: Account ID performing the operation
            provider: Vector DB provider name (default: "pinecone")
            batch_number: Optional batch UUID for grouping operations
        """
        self.account_id = account_id
        self.provider = provider
        self.batch_number = batch_number or str(uuid.uuid4())
        self._log_entry: Optional[VectorDbIngestionLog] = None
    
    def create_log_entry(
        self,
        operation_type: OperationType,
        index_name: str,
        namespace: Optional[str] = None,
        filenames: Optional[List[str]] = None,
        comment: Optional[str] = None,
        status: OperationStatus = OperationStatus.PENDING
    ) -> Optional[VectorDbIngestionLog]:
        """
        Create a new log entry in the database.
        
        Args:
            operation_type: Type of operation (INGEST, DELETE, UPDATE)
            index_name: Name of the vector database index
            namespace: Optional namespace
            filenames: List of filenames being processed
            comment: Optional comment
            status: Initial status (default: PENDING)
        
        Returns:
            Created VectorDbIngestionLog instance or None if failed
        """
        try:
            logger.info(f"Initializing database connection for logging (account_id={self.account_id})")
            init_database()
            
            db_gen = get_db()
            db = next(db_gen)
            
            try:
                logger.info(f"Creating VectorDbIngestionLog entry: account_id={self.account_id}, "
                          f"index_name={index_name}, namespace={namespace}, filenames={filenames}")
                
                log_entry = VectorDbIngestionLog(
                    operation_type=operation_type.value,
                    status=status.value,
                    account_id=self.account_id,
                    provider=self.provider,
                    index_name=index_name,
                    namespace=namespace,
                    filenames=filenames,
                    comment=comment,
                    batch_number=self.batch_number
                )
                
                db.add(log_entry)
                db.commit()
                db.refresh(log_entry)
                
                self._log_entry = log_entry
                logger.info(f"✅ Successfully created log entry {log_entry.id} for {operation_type.value} operation")
                
                return log_entry
            except Exception as e:
                db.rollback()
                logger.error(f"❌ Database transaction failed: {e}", exc_info=True)
                raise
            finally:
                try:
                    db.close()
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"❌ Failed to create log entry: {e}", exc_info=True)
            logger.error(f"   Account ID: {self.account_id}, Index: {index_name}, Namespace: {namespace}")
            # Don't raise - logging failures shouldn't break ingestion
            return None
    
    def update_log_entry(
        self,
        vectors_added: int = 0,
        vectors_deleted: int = 0,
        vectors_failed: int = 0,
        status: Optional[OperationStatus] = None,
        error_message: Optional[str] = None,
        error_code: Optional[str] = None,
        comment: Optional[str] = None
    ) -> bool:
        """
        Update an existing log entry.
        
        Args:
            vectors_added: Number of vectors successfully added
            vectors_deleted: Number of vectors deleted
            vectors_failed: Number of vectors that failed
            status: Updated status
            error_message: Error message if operation failed
            error_code: Error code if operation failed
            comment: Updated comment
        
        Returns:
            True if update was successful, False otherwise
        """
        if not self._log_entry:
            logger.warning("No log entry to update. Call create_log_entry first.")
            return False
        
        try:
            init_database()
            db_gen = get_db()
            db = next(db_gen)
            
            try:
                log_entry = db.query(VectorDbIngestionLog).filter(
                    VectorDbIngestionLog.id == self._log_entry.id
                ).first()
                
                if not log_entry:
                    logger.error(f"Log entry {self._log_entry.id} not found")
                    return False
                
                log_entry.vectors_added = vectors_added
                log_entry.vectors_deleted = vectors_deleted
                log_entry.vectors_failed = vectors_failed
                
                if status:
                    log_entry.status = status.value
                if error_message:
                    log_entry.error_message = error_message
                if error_code:
                    log_entry.error_code = error_code
                if comment:
                    log_entry.comment = comment
                
                db.commit()
                db.refresh(log_entry)
                
                logger.info(f"Updated log entry {log_entry.id} - Status: {log_entry.status}, "
                          f"Added: {vectors_added}, Failed: {vectors_failed}")
                
                return True
            except Exception as e:
                db.rollback()
                raise
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Failed to update log entry: {e}", exc_info=True)
            return False
    
    def log_success(
        self,
        vectors_added: int,
        vectors_failed: int = 0,
        comment: Optional[str] = None
    ) -> bool:
        """
        Log a successful operation.
        
        Args:
            vectors_added: Number of vectors successfully added
            vectors_failed: Number of vectors that failed
            comment: Optional comment
        
        Returns:
            True if logging was successful
        """
        final_status = OperationStatus.SUCCESS if vectors_failed == 0 else OperationStatus.PARTIAL
        return self.update_log_entry(
            vectors_added=vectors_added,
            vectors_failed=vectors_failed,
            status=final_status,
            comment=comment
        )
    
    def log_failure(
        self,
        error_message: str,
        error_code: Optional[str] = None,
        vectors_added: int = 0,
        vectors_failed: int = 0
    ) -> bool:
        """
        Log a failed operation.
        
        Args:
            error_message: Error message
            error_code: Optional error code
            vectors_added: Number of vectors added before failure
            vectors_failed: Number of vectors that failed
        
        Returns:
            True if logging was successful
        """
        return self.update_log_entry(
            vectors_added=vectors_added,
            vectors_failed=vectors_failed,
            status=OperationStatus.FAILED,
            error_message=error_message,
            error_code=error_code
        )

