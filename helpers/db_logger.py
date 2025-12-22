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
        self.log_entry_id: Optional[uuid.UUID] = None  # Track log entry ID separately for error handling
    
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
                self.log_entry_id = log_entry.id
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
        comment: Optional[str] = None,
        increment_counters: bool = True
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
        if not self.log_entry_id:
            logger.warning("No log entry ID available to update. Call create_log_entry first.")
            return False
        
        try:
            init_database()
            db_gen = get_db()
            db = next(db_gen)
            
            try:
                log_entry = db.query(VectorDbIngestionLog).filter(
                    VectorDbIngestionLog.id == self.log_entry_id
                ).first()
                
                if not log_entry:
                    logger.error(f"Log entry {self.log_entry_id} not found")
                    return False
                
                # Update counters - increment by default, or set absolute values if increment_counters=False
                if increment_counters:
                    log_entry.vectors_added = (log_entry.vectors_added or 0) + vectors_added
                    log_entry.vectors_deleted = (log_entry.vectors_deleted or 0) + vectors_deleted
                    log_entry.vectors_failed = (log_entry.vectors_failed or 0) + vectors_failed
                    logger.debug(f"Incremented counters - Added: +{vectors_added}, Failed: +{vectors_failed}")
                else:
                    # Set absolute values (for final status updates)
                    # Always set to the provided values - caller should pass the correct totals
                    log_entry.vectors_added = vectors_added
                    log_entry.vectors_deleted = vectors_deleted
                    log_entry.vectors_failed = vectors_failed
                    logger.info(f"Final status update - Setting absolute values: Added={vectors_added}, Failed={vectors_failed}")
                
                # Always update status if provided
                if status:
                    log_entry.status = status.value
                    logger.info(f"Setting status to: {status.value}")
                # If error_message is provided but no status, set status to FAILED
                elif error_message and log_entry.status == OperationStatus.PENDING.value:
                    log_entry.status = OperationStatus.FAILED.value
                    logger.info(f"Auto-setting status to FAILED due to error message")
                
                # Always update error fields if provided (for failures)
                if error_message:
                    # Append to existing error message if there is one
                    if log_entry.error_message:
                        log_entry.error_message = f"{log_entry.error_message}\n{error_message}"
                    else:
                        log_entry.error_message = error_message
                
                if error_code:
                    log_entry.error_code = error_code
                
                if comment:
                    log_entry.comment = comment
                
                db.commit()
                db.refresh(log_entry)
                
                logger.info(f"✅ Updated log entry {log_entry.id} - Status: {log_entry.status}, "
                          f"Added: {log_entry.vectors_added}, Failed: {log_entry.vectors_failed}, "
                          f"Error: {log_entry.error_message[:100] if log_entry.error_message else 'None'}")
                
                # Update cached log entry
                self._log_entry = log_entry
                return True
            except Exception as e:
                db.rollback()
                raise
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Failed to update log entry {self.log_entry_id}: {e}", exc_info=True)
            return False
    
    def log_error_safe(
        self,
        error_message: str,
        error_code: Optional[str] = None,
        vectors_added: int = 0,
        vectors_failed: int = 0,
        comment: Optional[str] = None
    ) -> bool:
        """
        Safely log an error, even if the log entry doesn't exist yet.
        This is a catch-all method that will create a log entry if needed.
        
        Args:
            error_message: Error message
            error_code: Optional error code
            vectors_added: Number of vectors added before failure
            vectors_failed: Number of vectors that failed
            comment: Optional comment
        
        Returns:
            True if logging was successful
        """
        try:
            # If we have a log entry ID, update it
            if self.log_entry_id:
                return self.log_failure(
                    error_message=error_message,
                    error_code=error_code,
                    vectors_added=vectors_added,
                    vectors_failed=vectors_failed
                )
            else:
                # Try to create a new log entry for this error
                logger.warning(f"Attempting to create log entry for error: {error_message[:100]}")
                try:
                    from db.enums import OperationType
                    log_entry = self.create_log_entry(
                        operation_type=OperationType.INGEST,
                        index_name="unknown",  # We don't know the index at this point
                        namespace=None,
                        filenames=None,
                        comment=comment or f"Error logged: {error_message[:200]}",
                        status=OperationStatus.FAILED
                    )
                    if log_entry:
                        return self.update_log_entry(
                            vectors_added=vectors_added,
                            vectors_failed=vectors_failed,
                            status=OperationStatus.FAILED,
                            error_message=error_message,
                            error_code=error_code
                        )
                except Exception as create_error:
                    logger.error(f"Failed to create log entry for error: {create_error}", exc_info=True)
                return False
        except Exception as e:
            logger.error(f"Failed to log error safely: {e}", exc_info=True)
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
        logger.info(f"Logging success: {vectors_added} added, {vectors_failed} failed, status: {final_status.value}")
        result = self.update_log_entry(
            vectors_added=vectors_added,
            vectors_failed=vectors_failed,
            status=final_status,
            comment=comment,
            increment_counters=False  # Set absolute values for final status
        )
        if result:
            logger.info(f"✅ Success status updated in database")
        else:
            logger.error(f"❌ Failed to update success status in database")
        return result
    
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
        logger.info(f"Logging failure: {error_message[:100]}, code: {error_code}, added: {vectors_added}, failed: {vectors_failed}")
        result = self.update_log_entry(
            vectors_added=vectors_added,
            vectors_failed=vectors_failed,
            status=OperationStatus.FAILED,  # Explicitly set status to FAILED
            error_message=error_message,
            error_code=error_code,
            increment_counters=False  # Set absolute values for final status
        )
        if result:
            logger.info(f"✅ Failure status updated in database")
        else:
            logger.error(f"❌ Failed to update failure status in database - log_entry_id: {self.log_entry_id}")
        return result

