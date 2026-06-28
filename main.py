import base64
import json
import logging
import sys
import time
from typing import Dict, Any, Union

from helpers.gcs import download_file_from_gcs, file_exists_in_gcs
from helpers.csv_processor import process_csv_file, process_qna_pairs
from helpers.pinecone import upsert_vectors, pinecone_api_key_for_account, ProcessingResult
from helpers.get_secret import get_secret
from helpers.db_logger import VectorDbLogger
from clients.gcs_client_factory import cloud_storage_client_for_account
from db.enums import OperationType, OperationStatus
from singletons.environment_variables import EnvironmentVariables

# IMPORTANT: In the Cloud Functions gen2 (Cloud Run) runtime, a root log handler
# is already installed before this module imports, so a plain basicConfig() call
# is a no-op and the root logger stays at WARNING. That silently drops every
# logger.info(...) in this function and its helpers — which is why upsert/ingest
# progress never showed up in Cloud Logging. force=True tears down the existing
# handlers and reconfigures at INFO so our diagnostics are actually emitted.
logging.basicConfig(
    level=logging.INFO,
    force=True,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def process_qna_ingest_topic_message(req_or_event: Union[Dict[str, Any], Any], context_or_res: Any = None):
    """
    Pub/Sub-triggered Cloud Function to process QnA CSV files.
    Also handles HTTP requests for local testing (matching Node.js implementation).
    
    Args:
        req_or_event: CloudEvent object (Pub/Sub) or Flask Request (HTTP)
        context_or_res: HTTP response context (for HTTP requests) or None
    
    Returns:
        ProcessingResult dictionary (Pub/Sub) or Flask response tuple (HTTP)
    """
    run_start = time.monotonic()
    message = "{}"
    attributes: Dict[str, str] = {}
    is_http_trigger = False
    db_logger = None  # Initialize for exception handler
    failed_rows_count = 0  # Track failed rows for exception handler
    successful_rows_count = 0  # Track successful rows for exception handler

    logger.info("===== QnA ingest function INVOKED =====")

    try:
        # Load secrets (with error handling in case Secret Manager is not accessible during import)
        try:
            EnvironmentVariables.EMBEDDINGS_API_URL = get_secret("EMBEDDINGS_API_URL")
            EnvironmentVariables.PINECONE_API_KEY = get_secret("PINECONE_API_KEY")
            EnvironmentVariables.PINECONE_ALL_MINILM_L6_V2_INDEX = get_secret("PINECONE_ALL_MINILM_L6_V2_INDEX")
            EnvironmentVariables.KB_INGEST_SA = get_secret("KB_INGEST_SA")
        except Exception as secret_error:
            logger.warning(f"Failed to load secrets during function execution: {secret_error}")
            # Try to load again (secrets might not be available during import but available at runtime)
            try:
                EnvironmentVariables.EMBEDDINGS_API_URL = get_secret("EMBEDDINGS_API_URL")
                EnvironmentVariables.PINECONE_API_KEY = get_secret("PINECONE_API_KEY")
                EnvironmentVariables.PINECONE_ALL_MINILM_L6_V2_INDEX = get_secret("PINECONE_ALL_MINILM_L6_V2_INDEX")
                EnvironmentVariables.KB_INGEST_SA = get_secret("KB_INGEST_SA")
            except Exception:
                raise Exception(f"Failed to load required secrets: {secret_error}")
        
        # Detect if triggered via HTTP or Pub/Sub (matching Node.js implementation)
        if hasattr(req_or_event, 'get_json') or (isinstance(req_or_event, dict) and 'body' in req_or_event):
            # HTTP-triggered (local testing with curl)
            is_http_trigger = True
            logger.info("Detected HTTP trigger for local testing.")
            
            if hasattr(req_or_event, 'get_json'):
                pub_sub_message = req_or_event.get_json() or {}
            else:
                pub_sub_message = req_or_event.get('body', {})
                if isinstance(pub_sub_message, str):
                    pub_sub_message = json.loads(pub_sub_message)
            
            logger.info(f"pubSubMessage: {pub_sub_message}")
            
            if 'data' in pub_sub_message:
                message = base64.b64decode(pub_sub_message['data']).decode('utf-8')
            else:
                message = json.dumps(pub_sub_message) if pub_sub_message else "{}"
            
            attributes = pub_sub_message.get('attributes', {})
            
            # Send HTTP response (matching Node.js behavior)
            if context_or_res and hasattr(context_or_res, 'status'):
                context_or_res.status(200).send("Message processed successfully.")
        else:
            # Pub/Sub-triggered (CloudEvent)
            cloud_event = req_or_event
            event_data = None
            
            if hasattr(cloud_event, 'data'):
                event_data = cloud_event.data
            elif isinstance(cloud_event, dict):
                event_data = cloud_event.get('data')
            
            # Pub/Sub message data is base64-encoded JSON string
            if event_data:
                if isinstance(event_data, bytes):
                    # Decode bytes to string, then decode base64
                    try:
                        decoded_str = event_data.decode('utf-8')
                        message = base64.b64decode(decoded_str).decode('utf-8')
                    except Exception:
                        # If not base64, treat as direct JSON
                        try:
                            message = event_data.decode('utf-8')
                        except Exception:
                            message = "{}"
                elif isinstance(event_data, str):
                    # Try to decode base64 first
                    try:
                        message = base64.b64decode(event_data).decode('utf-8')
                    except Exception:
                        # If not base64, use as-is
                        message = event_data
                elif isinstance(event_data, dict):
                    # If already a dict, serialize it
                    message = json.dumps(event_data)
                else:
                    message = str(event_data) if event_data else "{}"
            else:
                message = "{}"
            
            # Extract attributes from CloudEvent
            if hasattr(cloud_event, 'attributes'):
                attributes = dict(cloud_event.attributes) if cloud_event.attributes else {}
            elif isinstance(cloud_event, dict):
                attributes = cloud_event.get('attributes', {})
            else:
                attributes = {}
        
        # Parse the JSON message
        try:
            parsed_message: Dict[str, Any] = json.loads(message)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message as JSON: {message[:200]}")
            raise ValueError(f"Invalid JSON in message: {str(e)}")
        logger.info(f"Decoded message: {parsed_message}")
        # logger.info(f"Message attributes: {attributes}")
        
        file_id = parsed_message.get('file_id')
        filename = parsed_message.get('filename')
        gcs_bucket = parsed_message.get('gcs_bucket')
        gcs_file_path = parsed_message.get('gcs_file_path')
        file_size = parsed_message.get('file_size')
        content_type = parsed_message.get('content_type')
        user_id = parsed_message.get('user_id')
        user_email = parsed_message.get('user_email')
        namespace = parsed_message.get('namespace', 'similarity_search')
        # Target index chosen by the caller (e.g. the PDF-to-FAQ wizard). Fall
        # back to the configured default only when the message omits it, so
        # ingestion lands in the index the user actually selected.
        index_name = parsed_message.get('index_name') or EnvironmentVariables.PINECONE_ALL_MINILM_L6_V2_INDEX
        upload_timestamp = parsed_message.get('upload_timestamp')
        processing_status = parsed_message.get('processing_status')
        jwt = parsed_message.get('jwt')

        # PDF-to-FAQ flow: the reviewed Q&A pairs ride in the message and the
        # GCS file is the ORIGINAL PDF (the referenced source, not parsed here).
        source_type = parsed_message.get('source_type')
        qna_pairs = parsed_message.get('qna_pairs')
        is_pdf_faq = source_type == 'pdf_faq' and bool(qna_pairs)

        # Validate required fields
        if not filename or not gcs_bucket or not gcs_file_path:
            raise ValueError("Missing required fields: filename, gcs_bucket, or gcs_file_path")

        # Validate file type (CSV path only; the PDF-to-FAQ path stores a PDF)
        if not is_pdf_faq and not filename.lower().endswith('.csv'):
            raise ValueError("Only CSV files are supported")
        
        # Check if JWT is provided
        if not jwt:
            raise ValueError("JWT token is required for embedding API calls")

        # Resolve the account's own GCS credentials to download from its bucket.
        account_id_for_gcs = parsed_message.get('account_id')
        if account_id_for_gcs is None and user_id:
            try:
                account_id_for_gcs = int(user_id)
            except (TypeError, ValueError):
                account_id_for_gcs = None
        # Step 1: Obtain the Q&A content.
        # - CSV flow: download and parse the CSV from GCS.
        # - PDF-to-FAQ flow: the pairs are already in the message; the GCS file
        #   is the original PDF (the referenced source), so nothing to download.
        csv_content = None
        if is_pdf_faq:
            logger.info(f"Step 1: PDF-to-FAQ flow — using {len(qna_pairs)} Q&A pair(s) from message; "
                        f"original PDF referenced at gs://{gcs_bucket}/{gcs_file_path}")
        else:
            account_storage_client = cloud_storage_client_for_account(account_id_for_gcs)

            logger.info(f"Step 1: Downloading file from GCS: gs://{gcs_bucket}/{gcs_file_path}")
            try:
                # Check if file exists first
                file_exists = file_exists_in_gcs(gcs_bucket, gcs_file_path, account_storage_client)
                if not file_exists:
                    error_msg = f"File does not exist in GCS: gs://{gcs_bucket}/{gcs_file_path}"
                    logger.error(f"❌ {error_msg}")
                    if db_logger:
                        try:
                            db_logger.log_failure(
                                error_message=error_msg,
                                error_code="FileNotFoundError",
                                vectors_added=0,
                                vectors_failed=0
                            )
                        except Exception as log_error:
                            logger.warning(f"Failed to log GCS error to database: {log_error}")
                    raise FileNotFoundError(error_msg)

                csv_content = download_file_from_gcs(gcs_bucket, gcs_file_path, account_storage_client)

                if not csv_content.strip():
                    error_msg = "CSV file is empty"
                    logger.error(f"❌ {error_msg}")
                    if db_logger:
                        try:
                            db_logger.log_failure(
                                error_message=error_msg,
                                error_code="ValueError",
                                vectors_added=0,
                                vectors_failed=0
                            )
                        except Exception as log_error:
                            logger.warning(f"Failed to log empty file error to database: {log_error}")
                    raise ValueError(error_msg)
            except (FileNotFoundError, ValueError) as gcs_error:
                # Re-raise validation errors
                raise
            except Exception as gcs_error:
                error_msg = f"Failed to download file from GCS: {str(gcs_error)}"
                logger.error(f"❌ {error_msg}", exc_info=True)
                if db_logger:
                    try:
                        db_logger.log_failure(
                            error_message=error_msg,
                            error_code=type(gcs_error).__name__,
                            vectors_added=0,
                            vectors_failed=0
                        )
                    except Exception as log_error:
                        logger.warning(f"Failed to log GCS download error to database: {log_error}")
                raise Exception(error_msg) from gcs_error
        
        # Initialize database logger for tracking ingestion
        logger.info(f"Initializing database logger - user_id: {user_id}, user_email: {user_email}")
        try:
            account_id = int(user_id) if user_id else None
            logger.info(f"Parsed account_id: {account_id}")
            
            if not account_id:
                logger.warning(f"⚠️  Cannot create log entry: user_id is missing or invalid ({user_id})")
                db_logger = None
            else:
                logger.info(f"Creating VectorDbLogger for account_id={account_id}")
                db_logger = VectorDbLogger(
                    account_id=account_id,
                    provider="pinecone"
                )
                # Create log entry with PENDING status
                logger.info(f"🔵 [MAIN] Creating initial log entry with PENDING status...")
                logger.info(f"   Index: {index_name}")
                logger.info(f"   Namespace: {namespace}, Filename: {filename}")
                logger.info(f"   Account ID: {account_id}")

                log_entry = db_logger.create_log_entry(
                    operation_type=OperationType.INGEST,
                    index_name=index_name,
                    namespace=namespace,
                    filenames=[filename],
                    comment=f"Processing CSV file: {filename}",
                    status=OperationStatus.PENDING,
                    gcs_bucket=gcs_bucket,
                    gcs_file_path=gcs_file_path
                )
                
                logger.info(f"🔵 [MAIN] create_log_entry() returned: {log_entry}")
                logger.info(f"   db_logger.log_entry_id: {db_logger.log_entry_id}")
                
                if log_entry and db_logger.log_entry_id:
                    logger.info(f"✅ [MAIN] Database log entry created successfully: {log_entry.id}")
                    logger.info(f"   Status: {log_entry.status}")
                    logger.info(f"   Stored context - index: {db_logger._index_name}, namespace: {db_logger._namespace}")
                else:
                    logger.error(f"❌ [MAIN] CRITICAL: Log entry creation failed or log_entry_id not set!")
                    logger.error(f"   log_entry object: {log_entry}")
                    logger.error(f"   log_entry_id: {db_logger.log_entry_id if db_logger else None}")
                    logger.error(f"   This means the initial PENDING entry was NOT created")
                    logger.error(f"   Will attempt to create FAILED entry in catch-all handler")
                    # Don't set db_logger to None - keep it so we can try to create error entry later
                    # The context is already stored in db_logger (or will be stored manually)
        except Exception as db_error:
            logger.error(f"❌ Failed to initialize database logger: {db_error}", exc_info=True)
            # Try to create a logger anyway if we have account_id
            try:
                account_id = int(user_id) if user_id else None
                if account_id:
                    logger.info(f"Attempting to create db_logger after error: {account_id}")
                    db_logger = VectorDbLogger(account_id=account_id, provider="pinecone")
                    # Store context manually since create_log_entry failed
                    db_logger._index_name = index_name
                    db_logger._namespace = namespace
                    db_logger._filenames = [filename]
                else:
                    db_logger = None
            except Exception:
                db_logger = None

        # Resolve the account's OWN Pinecone API key up front, so ingestion targets
        # the same per-account Pinecone project the API microservice reads/manages
        # with (not a global one). Resolved by account_id reference + decrypted in
        # memory; the key is never placed in the Pub/Sub message or logged. Doing it
        # before Step 2 also fails fast (no wasted embedding work) if it's missing.
        pinecone_api_key = pinecone_api_key_for_account(account_id_for_gcs)

        # Step 2: Build vectors (parse CSV, or use Q&A pairs from the message)
        logger.info(f"Step 2: Processing source: {filename}")
        try:
            if is_pdf_faq:
                result_data = process_qna_pairs(
                    qna_pairs,
                    filename,
                    user_id,
                    user_email,
                    jwt,
                    db_logger,  # Pass db_logger for error tracking
                    gcs_bucket=gcs_bucket,
                    gcs_file_path=gcs_file_path
                )
            else:
                result_data = process_csv_file(
                    csv_content,
                    filename,
                    user_id,
                    user_email,
                    jwt,
                    db_logger,  # Pass db_logger for error tracking
                    gcs_bucket=gcs_bucket,
                    gcs_file_path=gcs_file_path
                )
        except Exception as csv_error:
            # Log CSV processing error to database
            if db_logger:
                try:
                    db_logger.log_failure(
                        error_message=f"CSV processing failed: {str(csv_error)}",
                        error_code=type(csv_error).__name__,
                        vectors_added=0,
                        vectors_failed=0
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log CSV error to database: {log_error}")
            raise
        
        vectors = result_data['vectors']
        successful_rows = result_data['successful_rows']
        failed_rows = result_data['failed_rows']

        # Store for exception handler
        successful_rows_count = successful_rows
        failed_rows_count = failed_rows

        logger.info(
            f"===== Step 2 complete: {successful_rows} embedding(s) built, "
            f"{failed_rows} failed; {len(vectors)} vector(s) ready to ingest ====="
        )

        # Step 3: Insert embeddings into Pinecone
        logger.info(
            f"===== Step 3: INGEST START — upserting {len(vectors)} vector(s) "
            f"into index '{index_name}', "
            f"namespace '{namespace}' ====="
        )
        if len(vectors) > 0:
            try:
                upsert_start = time.monotonic()
                upsert_vectors(vectors, namespace, index_name, pinecone_api_key)
                upsert_elapsed = time.monotonic() - upsert_start
                logger.info(
                    f"===== Step 3: INGEST COMPLETE — {len(vectors)} vector(s) upserted "
                    f"to '{index_name}'/"
                    f"'{namespace}' in {upsert_elapsed:.2f}s ====="
                )

                # Log success to database
                if db_logger:
                    try:
                        db_logger.log_success(
                            vectors_added=successful_rows,
                            vectors_failed=failed_rows,  # Use failed_rows from result_data
                            comment=f"Successfully ingested {successful_rows} vectors from {filename}"
                        )
                        logger.info(f"✅ Success logged to database: {successful_rows} vectors added, {failed_rows} failed")
                    except Exception as log_error:
                        logger.error(f"❌ Failed to log success to database: {log_error}", exc_info=True)
            except Exception as upsert_error:
                error_msg = f"Failed to upsert vectors to Pinecone: {str(upsert_error)}"
                logger.error(f"❌ {error_msg}", exc_info=True)
                
                # Log failure to database
                if db_logger:
                    try:
                        db_logger.log_failure(
                            error_message=error_msg,
                            error_code=type(upsert_error).__name__,
                            vectors_added=successful_rows,  # Count vectors that were generated successfully
                            vectors_failed=failed_rows + len(vectors)  # Failed during processing + failed to upsert
                        )
                        logger.info(f"✅ Failure logged to database: {error_msg[:100]}, {successful_rows} added, {failed_rows + len(vectors)} failed")
                    except Exception as log_error:
                        logger.error(f"❌ Failed to log Pinecone error to database: {log_error}", exc_info=True)
                raise Exception(error_msg) from upsert_error
        else:
            # No vectors to insert - log as failure
            error_msg = f"No vectors generated from CSV file. {failed_rows} row(s) failed to process."
            logger.warning(f"⚠️  {error_msg}")
            if db_logger:
                try:
                    logger.info(f"🔴 [MAIN] No vectors generated - calling log_failure()")
                    logger.info(f"   db_logger exists: {db_logger is not None}")
                    logger.info(f"   log_entry_id: {db_logger.log_entry_id}")
                    logger.info(f"   failed_rows: {failed_rows}")
                    logger.info(f"   error_msg: {error_msg}")
                    
                    # Use failed_rows from result_data - this includes all rows that failed during embedding generation
                    result = db_logger.log_failure(
                        error_message=error_msg,
                        error_code="NoVectorsGenerated",
                        vectors_added=0,
                        vectors_failed=failed_rows  # Use actual failed_rows count
                    )
                    
                    logger.info(f"🔴 [MAIN] log_failure() returned: {result}")
                    if result:
                        logger.info(f"✅ [MAIN] Failure logged to database successfully: {failed_rows} rows failed, 0 vectors added")
                    else:
                        logger.error(f"❌ [MAIN] log_failure() returned False - update may have failed")
                        logger.error(f"   This means the database update did not succeed")
                except Exception as log_error:
                    logger.error(f"❌ [MAIN] Exception calling log_failure(): {log_error}", exc_info=True)
                    logger.error(f"   Exception type: {type(log_error).__name__}")
                    logger.error(f"   db_logger.log_entry_id: {db_logger.log_entry_id if db_logger else None}")
            # Raise exception so the catch-all handler also logs this
            raise Exception(error_msg)
        
        # Prepare result
        result: ProcessingResult = {
            'success': True,
            'filename': filename,
            'total_chunks_created': successful_rows + failed_rows,
            'successful_uploads': successful_rows,
            'failed_uploads': failed_rows,
            'file_size_bytes': file_size
        }
        
        logger.info(
            f"===== DONE in {time.monotonic() - run_start:.2f}s — "
            f"completed successfully: {result} ====="
        )

        # Return appropriate response based on trigger type
        if is_http_trigger:
            from flask import jsonify
            return jsonify(result), 200
        else:
            return result
        
    except Exception as error:
        error_msg = str(error)
        error_code = type(error).__name__
        logger.error(
            f"❌ FAILED after {time.monotonic() - run_start:.2f}s — "
            f"{error_code}: {error_msg}",
            exc_info=True,
        )
        
        # Log failure to database - catch-all error handler
        try:
            logger.info(f"Catch-all handler - db_logger exists: {db_logger is not None}, "
                      f"log_entry_id: {db_logger.log_entry_id if db_logger else None}, "
                      f"failed_rows_count: {failed_rows_count}, successful_rows_count: {successful_rows_count}")
            
            # Try to log using db_logger if it exists and has a log entry
            if db_logger and db_logger.log_entry_id:
                try:
                    # Use the actual failed_rows_count if available, otherwise try to preserve existing count
                    vectors_failed_to_log = failed_rows_count if failed_rows_count > 0 else 0
                    vectors_added_to_log = successful_rows_count if successful_rows_count > 0 else 0
                    
                    logger.info(f"Updating existing log entry {db_logger.log_entry_id} with error: {error_code}, "
                              f"vectors_added={vectors_added_to_log}, vectors_failed={vectors_failed_to_log}")
                    
                    result = db_logger.log_failure(
                        error_message=error_msg,
                        error_code=error_code,
                        vectors_added=vectors_added_to_log,
                        vectors_failed=vectors_failed_to_log
                    )
                    if result:
                        logger.info(f"✅ Error logged to database successfully: {error_code}")
                    else:
                        logger.error(f"❌ log_failure() returned False - update may have failed")
                except Exception as log_error:
                    logger.error(f"❌ Failed to update log entry: {log_error}", exc_info=True)
            elif db_logger:
                logger.info(f"🔴 [MAIN] Path 2: db_logger exists BUT log_entry_id is None")
                logger.info(f"   This means initial log entry creation failed or log_entry_id was lost")
                logger.info(f"   Account ID: {db_logger.account_id}")
                logger.info(f"   Failed rows: {failed_rows_count}")
                logger.info(f"   Stored context - index: {db_logger._index_name}, namespace: {db_logger._namespace}")
                
                # Try to use safe error logging (creates entry if needed)
                # This happens when log_entry_id is None (initial entry creation failed)
                try:
                    logger.warning(f"⚠️  [MAIN] Using log_error_safe() to create new entry")
                    
                    result = db_logger.log_error_safe(
                        error_message=error_msg,
                        error_code=error_code,
                        vectors_added=successful_rows_count if successful_rows_count > 0 else 0,
                        vectors_failed=failed_rows_count if failed_rows_count > 0 else 0,
                        comment=f"Error during ingestion: {error_msg[:200]}"
                    )
                    
                    logger.info(f"🔴 [MAIN] log_error_safe() returned: {result}")
                    if result:
                        logger.info(f"✅ [MAIN] Error logged to database using safe method: {error_code}")
                    else:
                        logger.error(f"❌ [MAIN] log_error_safe() returned False")
                        logger.error(f"   This means error entry creation/update failed")
                except Exception as safe_log_error:
                    logger.error(f"❌ [MAIN] Exception calling log_error_safe(): {safe_log_error}", exc_info=True)
                    logger.error(f"   Exception type: {type(safe_log_error).__name__}")
                    logger.error(f"   Exception args: {safe_log_error.args}")
            else:
                # db_logger was never initialized - try to create one now for error logging
                logger.info(f"🔴 [MAIN] Path 3: db_logger is None - creating new logger for error logging")
                try:
                    account_id = int(user_id) if user_id else None
                    if account_id:
                        logger.info(f"🔴 [MAIN] Creating new VectorDbLogger for error logging (account_id={account_id})")
                        # VectorDbLogger is already imported at the top of the file
                        error_logger = VectorDbLogger(account_id=account_id, provider="pinecone")
                        error_logger.log_error_safe(
                            error_message=error_msg,
                            error_code=error_code,
                            vectors_added=0,
                            vectors_failed=0,
                            comment=f"Error during ingestion (logger created post-error): {error_msg[:200]}"
                        )
                        logger.info(f"✅ Error logged to database using post-error logger: {error_code}")
                except Exception as post_error_log_error:
                    logger.error(f"❌ Failed to create post-error logger: {post_error_log_error}", exc_info=True)
        except Exception as catch_all_log_error:
            logger.error(f"❌ Catch-all error logging failed: {catch_all_log_error}", exc_info=True)
        
        # Log the message that caused the error for debugging
        try:
            parsed_message = json.loads(message)
            logger.info(f"Decoded message: {parsed_message}")
        except Exception as parse_error:
            logger.info(f"Failed to parse message: {message}")
        
        # Return appropriate error response based on trigger type
        if is_http_trigger:
            from flask import jsonify
            error_result = {
                'success': False,
                'error': str(error)
            }
            return jsonify(error_result), 500
        else:
            raise Exception(f"Failed to process Pub/Sub message: {str(error)}")

