"""
CSV processing helper functions
"""
import csv
import hashlib
import logging
import io
from typing import List, Dict, Any, Optional
from helpers.embedding import fetch_embedding
from helpers.pinecone import VectorData

logger = logging.getLogger(__name__)


class CSVRow:
    """Represents a row from the CSV file."""
    def __init__(self, q: str, a: str):
        self.q = q
        self.a = a


class ProcessedRow:
    """Represents a processed row with metadata."""
    def __init__(
        self,
        question: str,
        answer: str,
        content: str,
        row_number: int,
        created_at: str,
        last_edited_at: str,
        uploaded_at: str
    ):
        self.question = question
        self.answer = answer
        self.content = content
        self.row_number = row_number
        self.created_at = created_at
        self.last_edited_at = last_edited_at
        self.uploaded_at = uploaded_at


def parse_csv_content(csv_content: str) -> List[ProcessedRow]:
    """
    Process CSV content and return processed rows.
    
    Args:
        csv_content: CSV file content as string
        
    Returns:
        List of ProcessedRow objects
    """
    rows: List[ProcessedRow] = []
    row_number = 0
    
    # Create a StringIO object to read CSV content
    csv_file = io.StringIO(csv_content)
    
    # Parse CSV using csv.DictReader
    reader = csv.DictReader(csv_file)
    
    for row_dict in reader:
        row_number += 1
        question = row_dict.get('q', '').strip() if row_dict.get('q') else ''
        answer = row_dict.get('a', '').strip() if row_dict.get('a') else ''
        created_at = row_dict.get('created_at', '').strip() if row_dict.get('created_at') else ''
        last_edited_at = row_dict.get('last_edited_at', '').strip() if row_dict.get('last_edited_at') else ''
        
        if question and answer:
            rows.append(ProcessedRow(
                question=question,
                answer=answer,
                content=f"Q: {question}\nA: {answer}",
                row_number=row_number,
                created_at=created_at,
                last_edited_at=last_edited_at,
                uploaded_at=str(int(__import__('time').time() * 1000))  # Milliseconds timestamp
            ))
        else:
            logger.info(f'Skipping row {row_number}: empty question or answer')
    
    return rows


def generate_embedding_for_row(
    row: ProcessedRow,
    filename: str,
    user_id: str,
    user_email: str,
    jwt: str,
    db_logger=None,
    gcs_bucket: Optional[str] = None,
    gcs_file_path: Optional[str] = None
) -> Optional[VectorData]:
    """
    Generate embedding for a single row and prepare vector data.
    
    Args:
        row: ProcessedRow object
        filename: Name of the CSV file
        user_id: User ID
        user_email: User email
        jwt: JWT token for embedding API
        db_logger: Optional VectorDbLogger instance for error logging
        
    Returns:
        VectorData dictionary or None if embedding generation fails
    """
    try:

        print('--- --- --- --- ---')
        print(f'Generating embedding for row {row.row_number}')
        print('--- --- --- --- ---')
        logger.info(f'--- --- --- --- ---')
        logger.info(f'Generating embedding for row {row.row_number}')
        logger.info(f'--- --- --- --- ---')

        # Generate embedding for the content
        embedding = fetch_embedding(jwt, row.content)
        
        if not embedding or len(embedding) == 0:
            logger.info(f'Failed to generate embedding for row {row.row_number}')
            return None
        
        # Validate embedding values are all numbers
        validated_embedding = []
        for index, val in enumerate(embedding):
            try:
                num = float(val)
                if num != num:  # Check for NaN
                    raise ValueError(f'Invalid embedding value at index {index}: {val}')
                validated_embedding.append(num)
            except (ValueError, TypeError) as e:
                raise ValueError(f'Invalid embedding value at index {index}: {val}') from e
        
        logger.info(
            f'Row {row.row_number}: Generated embedding with {len(validated_embedding)} dimensions'
        )
        
        # Create unique ID for the vector
        id_content = f'{filename}_{row.row_number}_{row.question[:50]}'
        chunk_id = hashlib.sha256(id_content.encode()).hexdigest()
        
        # Prepare metadata
        metadata = {
            'row_number': row.row_number,
            'q': row.question,
            'a': row.answer,
            'content': row.content,
            'filename': filename,
            'user_id': user_id,
            'user_email': user_email,
            'upload_timestamp': row.uploaded_at,
            'created_at': row.created_at,
            'last_edited_at': row.last_edited_at,
            # Provider-agnostic pointer back to the original source document.
            'storage_provider': 'gcs' if (gcs_bucket or gcs_file_path) else None,
            'storage_bucket': gcs_bucket,
            'storage_path': gcs_file_path,
        }
        
        # Filter out None values (Pinecone doesn't accept null metadata values)
        metadata = {k: v for k, v in metadata.items() if v is not None}
        
        # Prepare vector data
        vector_data: VectorData = {
            'id': chunk_id,
            'values': validated_embedding,
            'metadata': metadata,
        }
        
        return vector_data
    except Exception as error:
        error_msg = f'Error processing row {row.row_number}: {str(error)}'
        logger.error(f'❌ {error_msg}', exc_info=True)
        
        # Log error to database if logger is available
        if db_logger:
            try:
                # Increment failed vectors count and update status to PARTIAL if not already FAILED
                # Note: We don't set status to FAILED here because the overall operation might still succeed
                # The final status will be set in main.py based on overall success/failure
                db_logger.update_log_entry(
                    vectors_failed=1,
                    error_message=error_msg,
                    error_code=type(error).__name__
                )
                logger.info(f"✅ Logged row {row.row_number} error to database")
            except Exception as log_error:
                logger.error(f'❌ Failed to log row error to database: {log_error}', exc_info=True)
        
        return None


def process_qna_pairs(
    qna_pairs: List[Dict[str, Any]],
    filename: str,
    user_id: str,
    user_email: str,
    jwt: str,
    db_logger=None,
    gcs_bucket: Optional[str] = None,
    gcs_file_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate embeddings for pre-reviewed Q&A pairs delivered in the Pub/Sub
    message (the PDF-to-FAQ flow).

    Mirrors process_csv_file but skips CSV parsing — the pairs are already
    structured. `filename`/`gcs_bucket`/`gcs_file_path` point at the ORIGINAL
    PDF stored in GCS, so each vector's source metadata references the PDF.

    Returns:
        Dictionary with 'vectors', 'successful_rows', and 'failed_rows'
    """
    try:
        uploaded_at = str(int(__import__('time').time() * 1000))
        rows: List[ProcessedRow] = []
        for index, pair in enumerate(qna_pairs, start=1):
            question = str(pair.get('q', '')).strip() if pair.get('q') else ''
            answer = str(pair.get('a', '')).strip() if pair.get('a') else ''
            if question and answer:
                rows.append(ProcessedRow(
                    question=question,
                    answer=answer,
                    content=f"Q: {question}\nA: {answer}",
                    row_number=index,
                    created_at='',
                    last_edited_at='',
                    uploaded_at=uploaded_at
                ))
            else:
                logger.info(f'Skipping Q&A pair {index}: empty question or answer')

        if len(rows) == 0:
            raise ValueError("No valid Q&A pairs found in message")

        logger.info(f'Processing {len(rows)} Q&A pairs from source: {filename}')

        vectors: List[VectorData] = []
        successful_rows = 0
        failed_rows = 0

        for row in rows:
            vector_data = generate_embedding_for_row(
                row,
                filename,
                user_id,
                user_email,
                jwt,
                db_logger,
                gcs_bucket=gcs_bucket,
                gcs_file_path=gcs_file_path
            )

            if vector_data:
                vectors.append(vector_data)
                successful_rows += 1
            else:
                failed_rows += 1

        logger.info(
            f'Successfully processed {successful_rows} pairs, failed {failed_rows} pairs'
        )

        return {
            'vectors': vectors,
            'successful_rows': successful_rows,
            'failed_rows': failed_rows,
        }
    except Exception as error:
        error_msg = f'Error processing Q&A pairs: {str(error)}'
        logger.error(f'❌ {error_msg}', exc_info=True)

        if db_logger:
            try:
                db_logger.log_failure(
                    error_message=error_msg,
                    error_code=type(error).__name__,
                    vectors_added=0,
                    vectors_failed=0
                )
            except Exception as log_error:
                logger.warning(f'Failed to log Q&A processing error to database: {log_error}')

        raise Exception(error_msg) from error


def process_csv_file(
    csv_content: str,
    filename: str,
    user_id: str,
    user_email: str,
    jwt: str,
    db_logger=None,
    gcs_bucket: Optional[str] = None,
    gcs_file_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process CSV file and generate embeddings for all rows.
    
    Args:
        csv_content: CSV file content as string
        filename: Name of the CSV file
        user_id: User ID
        user_email: User email
        jwt: JWT token for embedding API
        db_logger: Optional VectorDbLogger instance for error logging
        
    Returns:
        Dictionary with 'vectors', 'successful_rows', and 'failed_rows'
    """
    try:
        # Parse CSV content
        rows = parse_csv_content(csv_content)
        
        if len(rows) == 0:
            raise ValueError("No valid rows found in CSV file")
        
        logger.info(f'Processing {len(rows)} rows from CSV file: {filename}')
        
        # Process each row and generate embeddings
        vectors: List[VectorData] = []
        successful_rows = 0
        failed_rows = 0
        
        for row in rows:
            logger.info(
                f'Processing row {row.row_number}: {row.question[:10]}'
            )
            
            vector_data = generate_embedding_for_row(
                row,
                filename,
                user_id,
                user_email,
                jwt,
                db_logger,
                gcs_bucket=gcs_bucket,
                gcs_file_path=gcs_file_path
            )
            
            if vector_data:
                vectors.append(vector_data)
                successful_rows += 1
            else:
                failed_rows += 1
        
        logger.info(
            f'Successfully processed {successful_rows} rows, failed {failed_rows} rows'
        )
        
        return {
            'vectors': vectors,
            'successful_rows': successful_rows,
            'failed_rows': failed_rows,
        }
    except Exception as error:
        error_msg = f'Error processing CSV file: {str(error)}'
        logger.error(f'❌ {error_msg}', exc_info=True)
        
        # Log error to database if logger is available
        if db_logger:
            try:
                db_logger.log_failure(
                    error_message=error_msg,
                    error_code=type(error).__name__,
                    vectors_added=successful_rows,
                    vectors_failed=failed_rows
                )
            except Exception as log_error:
                logger.warning(f'Failed to log CSV processing error to database: {log_error}')
        
        raise Exception(error_msg) from error

