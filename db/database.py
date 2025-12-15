"""
Database connection and session management.
Shared with FastAPI microservice - uses same models and connection pattern.
"""
import logging
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator
from helpers.get_secret import get_secret

logger = logging.getLogger(__name__)

Base = declarative_base()

# Global session factory (will be initialized on first use)
SessionLocal: sessionmaker = None
engine = None


def get_database_url() -> str:
    """
    Get database URL from Secret Manager or environment variable.
    
    Returns:
        Database connection URL string
    """
    try:
        # Try to get from Secret Manager first
        logger.info("Attempting to get POSTGRES_URL from Secret Manager...")
        db_url = get_secret("POSTGRES_URL")
        if db_url:
            logger.info(f"✅ Retrieved POSTGRES_URL from Secret Manager (length: {len(db_url)})")
            return db_url
    except Exception as e:
        logger.warning(f"⚠️  Failed to get POSTGRES_URL from Secret Manager: {e}")
    
    # Fallback to environment variable (for local development)
    import os
    db_url = os.getenv("POSTGRES_URL")
    if db_url:
        logger.info(f"✅ Using POSTGRES_URL from environment variable (length: {len(db_url)})")
        return db_url
    
    error_msg = "POSTGRES_URL not found in Secret Manager or environment variables"
    logger.error(f"❌ {error_msg}")
    raise ValueError(error_msg)


def init_database():
    """
    Initialize database connection and session factory.
    Should be called before using the database.
    """
    global SessionLocal, engine
    
    if engine is not None:
        logger.debug("Database already initialized, skipping")
        return  # Already initialized
    
    try:
        logger.info("Initializing database connection...")
        database_url = get_database_url()
        
        # Mask password in URL for logging
        safe_url = database_url
        if '@' in database_url:
            parts = database_url.split('@')
            if len(parts) == 2:
                safe_url = '***@' + parts[1]
        
        logger.info(f"Creating engine with URL: {safe_url}")
        engine = create_engine(
            database_url,
            pool_pre_ping=True,  # Verify connections before using
            pool_size=5,
            max_overflow=10,
            echo=False  # Set to True for SQL query logging
        )
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        logger.info("✅ Database connection initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize database connection: {e}", exc_info=True)
        raise


def get_db() -> Generator[Session, None, None]:
    """
    Get database session.
    Use as a context manager or dependency injection.
    
    Yields:
        SQLAlchemy Session
    """
    if SessionLocal is None:
        init_database()
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

