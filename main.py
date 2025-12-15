import base64
import json
import logging
from typing import Dict, Any, Union

from helpers.gcs import download_file_from_gcs, file_exists_in_gcs
from helpers.csv_processor import process_csv_file
from helpers.pinecone import upsert_vectors, ProcessingResult
from helpers.get_secret import get_secret
from singletons.environment_variables import EnvironmentVariables

logging.basicConfig(level=logging.INFO)
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
    message = "{}"
    attributes: Dict[str, str] = {}
    is_http_trigger = False
    
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
        upload_timestamp = parsed_message.get('upload_timestamp')
        processing_status = parsed_message.get('processing_status')
        jwt = parsed_message.get('jwt')
        
        # Validate required fields
        if not filename or not gcs_bucket or not gcs_file_path:
            raise ValueError("Missing required fields: filename, gcs_bucket, or gcs_file_path")
        
        # Validate file type (only CSV files are supported)
        if not filename.lower().endswith('.csv'):
            raise ValueError("Only CSV files are supported")
        
        # Check if JWT is provided
        if not jwt:
            raise ValueError("JWT token is required for embedding API calls")
        
        # Step 1: Download file from GCS
        logger.info(f"Step 1: Downloading file from GCS: gs://{gcs_bucket}/{gcs_file_path}")
        
        # Check if file exists first
        file_exists = file_exists_in_gcs(gcs_bucket, gcs_file_path)
        if not file_exists:
            raise FileNotFoundError(f"File does not exist in GCS: gs://{gcs_bucket}/{gcs_file_path}")
        
        csv_content = download_file_from_gcs(gcs_bucket, gcs_file_path)
        
        if not csv_content.strip():
            raise ValueError("CSV file is empty")
        
        # Step 2: Parse CSV and generate embeddings
        logger.info(f"Step 2: Processing CSV file: {filename}")
        result_data = process_csv_file(
            csv_content,
            filename,
            user_id,
            user_email,
            jwt
        )
        
        vectors = result_data['vectors']
        successful_rows = result_data['successful_rows']
        failed_rows = result_data['failed_rows']
        
        # Step 3: Insert embeddings into Pinecone
        logger.info(f"Step 3: Inserting {len(vectors)} vectors into Pinecone")
        if len(vectors) > 0:
            upsert_vectors(vectors, namespace)
        
        # Prepare result
        result: ProcessingResult = {
            'success': True,
            'filename': filename,
            'total_chunks_created': successful_rows + failed_rows,
            'successful_uploads': successful_rows,
            'failed_uploads': failed_rows,
            'file_size_bytes': file_size
        }
        
        logger.info(f"Processing completed successfully: {result}")
        
        # Return appropriate response based on trigger type
        if is_http_trigger:
            from flask import jsonify
            return jsonify(result), 200
        else:
            return result
        
    except Exception as error:
        logger.error(f"Error processing message: {error}", exc_info=True)
        
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

