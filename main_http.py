"""
HTTP-triggered version for local testing
"""
import base64
import json
import logging
from typing import Dict, Any
from flask import Request

from helpers.gcs import download_file_from_gcs, file_exists_in_gcs
from helpers.csv_processor import process_csv_file
from helpers.pinecone import upsert_vectors, ProcessingResult
from helpers.get_secret import get_secret
from singletons.environment_variables import EnvironmentVariables

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_qna_ingest_http(request: Request):
    """
    HTTP-triggered Cloud Function for local testing.
    
    Args:
        request: Flask request object
        
    Returns:
        Response string
    """
    message = "{}"
    attributes: Dict[str, str] = {}
    
    try:
        # Load secrets
        EnvironmentVariables.EMBEDDINGS_API_URL = get_secret("EMBEDDINGS_API_URL")
        EnvironmentVariables.PINECONE_API_KEY = get_secret("PINECONE_API_KEY")
        EnvironmentVariables.PINECONE_ALL_MINILM_L6_V2_INDEX = get_secret("PINECONE_ALL_MINILM_L6_V2_INDEX")
        EnvironmentVariables.KB_INGEST_SA = get_secret("KB_INGEST_SA")
        
        # HTTP-triggered (local testing with curl)
        logger.info("Detected HTTP trigger for local testing.")
        pub_sub_message = request.get_json() or {}
        logger.info(f"pubSubMessage: {pub_sub_message}")
        
        if 'data' in pub_sub_message:
            message = base64.b64decode(pub_sub_message['data']).decode('utf-8')
        else:
            message = json.dumps(pub_sub_message) if pub_sub_message else "{}"
        
        attributes = pub_sub_message.get('attributes', {})
        
        # Decode and process the message
        parsed_message: Dict[str, Any] = json.loads(message)
        logger.info(f"Decoded message: {parsed_message}")
        logger.info(f"Message attributes: {attributes}")
        
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
        
        from flask import jsonify
        return jsonify(result), 200
        
    except Exception as error:
        logger.error(f"Error processing HTTP request: {error}", exc_info=True)
        
        # Log the message that caused the error for debugging
        try:
            parsed_message = json.loads(message)
            logger.info(f"Decoded message: {parsed_message}")
            logger.info(f"Message attributes: {attributes}")
        except Exception as parse_error:
            logger.info(f"Failed to parse message: {message}")
        
        # Return error response
        from flask import jsonify
        error_result = {
            'success': False,
            'error': str(error)
        }
        return jsonify(error_result), 500

