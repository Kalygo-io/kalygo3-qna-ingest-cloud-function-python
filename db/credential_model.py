"""
Read-only Credential model mirroring the API microservice's credentials table.

Used by the ingest function to resolve an account's per-account Google Cloud
Storage service-account credential. Migrations are owned by the API microservice;
this service only reads.
"""
from sqlalchemy import Column, Integer, String, Text, JSON, DateTime, func
from sqlalchemy import Enum as SAEnum
from .database import Base
from .enums import CredentialType


class Credential(Base):
    __tablename__ = 'credentials'

    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, nullable=False, index=True)
    # Native PG enum credential_type_enum (created by the API microservice).
    credential_type = Column(
        SAEnum(CredentialType, name='credential_type_enum', create_type=False),
        nullable=False,
        index=True,
    )
    auth_type = Column(String(50), nullable=True)
    credential_name = Column(String(255), nullable=True)
    encrypted_data = Column(Text, nullable=False)
    credential_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now())
