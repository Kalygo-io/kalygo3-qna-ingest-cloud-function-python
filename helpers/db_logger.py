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
        # Store context for error logging if initial entry creation fails
        self._index_name: Optional[str] = None
        self._namespace: Optional[str] = None
        self._filenames: Optional[List[str]] = None
    
    def create_log_entry(
        self,
        operation_type: OperationType,
        index_name: str,
        namespace: Optional[str] = None,
        filenames: Optional[List[str]] = None,
        comment: Optional[str] = None,
        status: OperationStatus = OperationStatus.PENDING,
        gcs_bucket: Optional[str] = None,
        gcs_file_path: Optional[str] = None
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
            logger.info(f"🔵 [DB_LOGGER] Initializing database connection for logging")
            logger.info(f"   Account ID: {self.account_id}, Index: {index_name}, Namespace: {namespace}")
            logger.info(f"   Filenames: {filenames}, Status: {status.value}, Operation: {operation_type.value}")
            
            init_database()
            logger.info(f"🔵 [DB_LOGGER] Database initialized, getting session...")
            
            db_gen = get_db()
            db = next(db_gen)
            logger.info(f"🔵 [DB_LOGGER] Database session obtained")
            
            try:
                logger.info(f"🔵 [DB_LOGGER] Creating VectorDbIngestionLog entry...")
                
                log_entry = VectorDbIngestionLog(
                    operation_type=operation_type.value,
                    status=status.value,
                    account_id=self.account_id,
                    provider=self.provider,
                    index_name=index_name,
                    namespace=namespace,
                    filenames=filenames,
                    comment=comment,
                    gcs_bucket=gcs_bucket,
                    gcs_file_path=gcs_file_path,
                    batch_number=self.batch_number
                )
                
                logger.info(f"🔵 [DB_LOGGER] Log entry object created, adding to session...")
                db.add(log_entry)
                logger.info(f"🔵 [DB_LOGGER] Committing transaction...")
                db.commit()
                logger.info(f"🔵 [DB_LOGGER] Transaction committed, refreshing object...")
                db.refresh(log_entry)
                
                self._log_entry = log_entry
                self.log_entry_id = log_entry.id
                # Store context for potential error logging
                self._index_name = index_name
                self._namespace = namespace
                self._filenames = filenames
                
                logger.info(f"✅ [DB_LOGGER] Successfully created log entry!")
                logger.info(f"   Log Entry ID: {log_entry.id}")
                logger.info(f"   self.log_entry_id: {self.log_entry_id}")
                logger.info(f"   Status: {log_entry.status}")
                logger.info(f"   Stored context - index: {self._index_name}, namespace: {self._namespace}")
                
                return log_entry
            except Exception as e:
                logger.error(f"❌ [DB_LOGGER] Database transaction failed: {e}", exc_info=True)
                logger.error(f"   Exception type: {type(e).__name__}")
                logger.error(f"   Exception args: {e.args}")
                try:
                    db.rollback()
                    logger.info(f"🔵 [DB_LOGGER] Transaction rolled back")
                except Exception as rollback_error:
                    logger.error(f"❌ [DB_LOGGER] Failed to rollback: {rollback_error}")
                raise
            finally:
                try:
                    db.close()
                    logger.info(f"🔵 [DB_LOGGER] Database session closed")
                except Exception as close_error:
                    logger.warning(f"⚠️ [DB_LOGGER] Error closing session: {close_error}")
        except Exception as e:
            logger.error(f"❌ [DB_LOGGER] Failed to create log entry: {e}", exc_info=True)
            logger.error(f"   Exception type: {type(e).__name__}")
            logger.error(f"   Account ID: {self.account_id}, Index: {index_name}, Namespace: {namespace}")
            logger.error(f"   self.log_entry_id after failure: {self.log_entry_id}")
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
        logger.info(f"🔵 [DB_LOGGER] update_log_entry() called")
        logger.info(f"   log_entry_id: {self.log_entry_id}")
        logger.info(f"   Parameters - vectors_added: {vectors_added}, vectors_failed: {vectors_failed}")
        logger.info(f"   status: {status.value if status else None}, error_code: {error_code}")
        logger.info(f"   increment_counters: {increment_counters}")
        logger.info(f"   error_message length: {len(error_message) if error_message else 0}")
        
        if not self.log_entry_id:
            logger.error(f"❌ [DB_LOGGER] No log entry ID available to update!")
            logger.error(f"   self.log_entry_id: {self.log_entry_id}")
            logger.error(f"   self._log_entry: {self._log_entry}")
            return False
        
        try:
            logger.info(f"🔵 [DB_LOGGER] Initializing database for update...")
            init_database()
            logger.info(f"🔵 [DB_LOGGER] Getting database session...")
            db_gen = get_db()
            db = next(db_gen)
            logger.info(f"🔵 [DB_LOGGER] Database session obtained")
            
            try:
                logger.info(f"🔵 [DB_LOGGER] Querying for log entry with ID: {self.log_entry_id}")
                log_entry = db.query(VectorDbIngestionLog).filter(
                    VectorDbIngestionLog.id == self.log_entry_id
                ).first()
                
                if not log_entry:
                    logger.error(f"❌ [DB_LOGGER] Log entry {self.log_entry_id} not found in database!")
                    logger.error(f"   Query returned None - entry may have been deleted or ID is incorrect")
                    return False
                
                logger.info(f"✅ [DB_LOGGER] Log entry found in database")
                # Log current state before update
                logger.info(f"🔵 [DB_LOGGER] Current log entry state BEFORE update:")
                logger.info(f"   Status: {log_entry.status}")
                logger.info(f"   Vectors Added: {log_entry.vectors_added}")
                logger.info(f"   Vectors Failed: {log_entry.vectors_failed}")
                logger.info(f"   Error Message: {log_entry.error_message[:100] if log_entry.error_message else None}")
                logger.info(f"   Error Code: {log_entry.error_code}")
                
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
                    logger.info(f"✅ Setting status to: {status.value}")
                # If error_message is provided but no status, set status to FAILED (unless already set)
                elif error_message:
                    if log_entry.status == OperationStatus.PENDING.value:
                        log_entry.status = OperationStatus.FAILED.value
                        logger.info(f"✅ Auto-setting status to FAILED due to error message")
                    else:
                        logger.info(f"Status already set to {log_entry.status}, not changing to FAILED")
                
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
                
                logger.info(f"🔵 [DB_LOGGER] Prepared updates, committing transaction...")
                logger.info(f"   New Status: {log_entry.status}")
                logger.info(f"   New Vectors Added: {log_entry.vectors_added}")
                logger.info(f"   New Vectors Failed: {log_entry.vectors_failed}")
                
                db.commit()
                logger.info(f"✅ [DB_LOGGER] Transaction committed successfully")
                
                db.refresh(log_entry)
                logger.info(f"🔵 [DB_LOGGER] Object refreshed from database")
                
                logger.info(f"✅ [DB_LOGGER] Updated log entry {log_entry.id}")
                logger.info(f"   Final Status: {log_entry.status}")
                logger.info(f"   Final Vectors Added: {log_entry.vectors_added}")
                logger.info(f"   Final Vectors Failed: {log_entry.vectors_failed}")
                logger.info(f"   Final Error Message: {log_entry.error_message[:200] if log_entry.error_message else 'None'}")
                logger.info(f"   Final Error Code: {log_entry.error_code}")
                
                # Update cached log entry
                self._log_entry = log_entry
                return True
            except Exception as e:
                logger.error(f"❌ [DB_LOGGER] Exception during update transaction: {e}", exc_info=True)
                logger.error(f"   Exception type: {type(e).__name__}")
                logger.error(f"   Exception args: {e.args}")
                try:
                    db.rollback()
                    logger.info(f"🔵 [DB_LOGGER] Transaction rolled back")
                except Exception as rollback_error:
                    logger.error(f"❌ [DB_LOGGER] Failed to rollback: {rollback_error}")
                raise
            finally:
                try:
                    db.close()
                    logger.info(f"🔵 [DB_LOGGER] Database session closed")
                except Exception as close_error:
                    logger.warning(f"⚠️ [DB_LOGGER] Error closing session: {close_error}")
        except Exception as e:
            logger.error(f"❌ [DB_LOGGER] Failed to update log entry {self.log_entry_id}: {e}", exc_info=True)
            logger.error(f"   Exception type: {type(e).__name__}")
            logger.error(f"   Exception args: {e.args}")
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
        logger.info(f"🟡 [DB_LOGGER] log_error_safe() called")
        logger.info(f"   log_entry_id: {self.log_entry_id}")
        logger.info(f"   error_message: {error_message[:200]}")
        logger.info(f"   error_code: {error_code}")
        logger.info(f"   vectors_added: {vectors_added}, vectors_failed: {vectors_failed}")
        
        try:
            # If we have a log entry ID, update it
            if self.log_entry_id:
                logger.info(f"🟡 [DB_LOGGER] log_entry_id exists ({self.log_entry_id}), calling log_failure()...")
                result = self.log_failure(
                    error_message=error_message,
                    error_code=error_code,
                    vectors_added=vectors_added,
                    vectors_failed=vectors_failed
                )
                logger.info(f"🟡 [DB_LOGGER] log_failure() returned: {result}")
                return result
            else:
                # Try to create a new log entry for this error
                logger.warning(f"⚠️  [DB_LOGGER] No existing log entry found (log_entry_id is None)")
                logger.warning(f"   Attempting to create log entry for error: {error_message[:100]}")
                logger.info(f"   Stored context - index: {self._index_name}, namespace: {self._namespace}, filenames: {self._filenames}")
                
                try:
                    from db.enums import OperationType
                    # Use stored context if available, otherwise use defaults
                    index_name = self._index_name or "unknown"
                    namespace = self._namespace
                    filenames = self._filenames
                    
                    logger.info(f"🟡 [DB_LOGGER] Creating error log entry with context:")
                    logger.info(f"   index: {index_name}, namespace: {namespace}, filenames: {filenames}")
                    
                    logger.info(f"🟡 [DB_LOGGER] Calling create_log_entry() with FAILED status...")
                    log_entry = self.create_log_entry(
                        operation_type=OperationType.INGEST,
                        index_name=index_name,
                        namespace=namespace,
                        filenames=filenames,
                        comment=comment or f"Error logged: {error_message[:200]}",
                        status=OperationStatus.FAILED  # Create with FAILED status directly
                    )
                    
                    logger.info(f"🟡 [DB_LOGGER] create_log_entry() returned: {log_entry}")
                    logger.info(f"   self.log_entry_id after create: {self.log_entry_id}")
                    
                    if log_entry and self.log_entry_id:
                        logger.info(f"✅ [DB_LOGGER] Created error log entry {self.log_entry_id}")
                        logger.info(f"   Entry status: {log_entry.status}")
                        logger.info(f"   Now updating with error details...")
                        
                        # Update with error details and vector counts
                        result = self.update_log_entry(
                            vectors_added=vectors_added,
                            vectors_failed=vectors_failed,
                            status=OperationStatus.FAILED,  # Ensure status is FAILED
                            error_message=error_message,
                            error_code=error_code,
                            increment_counters=False
                        )
                        logger.info(f"🟡 [DB_LOGGER] update_log_entry() returned: {result}")
                        
                        if result:
                            logger.info(f"✅ [DB_LOGGER] Error log entry successfully created and updated")
                        else:
                            logger.error(f"❌ [DB_LOGGER] Error log entry created but update failed!")
                        
                        return result
                    else:
                        logger.error(f"❌ [DB_LOGGER] Failed to create error log entry or log_entry_id not set")
                        logger.error(f"   log_entry: {log_entry}")
                        logger.error(f"   log_entry_id: {self.log_entry_id}")
                        logger.error(f"   This means create_log_entry() failed silently")
                        return False
                except Exception as create_error:
                    logger.error(f"❌ [DB_LOGGER] Exception creating log entry for error: {create_error}", exc_info=True)
                    logger.error(f"   Exception type: {type(create_error).__name__}")
                    return False
        except Exception as e:
            logger.error(f"❌ [DB_LOGGER] Exception in log_error_safe(): {e}", exc_info=True)
            logger.error(f"   Exception type: {type(e).__name__}")
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
        logger.info(f"🔴 [DB_LOGGER] log_failure() called")
        logger.info(f"   log_entry_id: {self.log_entry_id}")
        logger.info(f"   error_message: {error_message[:200]}")
        logger.info(f"   error_code: {error_code}")
        logger.info(f"   vectors_added: {vectors_added}, vectors_failed: {vectors_failed}")
        
        if not self.log_entry_id:
            logger.error(f"❌ [DB_LOGGER] log_failure() called but log_entry_id is None!")
            logger.error(f"   Cannot update non-existent log entry")
            return False
        
        result = self.update_log_entry(
            vectors_added=vectors_added,
            vectors_failed=vectors_failed,
            status=OperationStatus.FAILED,  # Explicitly set status to FAILED
            error_message=error_message,
            error_code=error_code,
            increment_counters=False  # Set absolute values for final status
        )
        if result:
            logger.info(f"✅ [DB_LOGGER] Failure status updated in database successfully")
        else:
            logger.error(f"❌ [DB_LOGGER] Failed to update failure status in database!")
            logger.error(f"   log_entry_id: {self.log_entry_id}")
            logger.error(f"   update_log_entry() returned False")
        return result

