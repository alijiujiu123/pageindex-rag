"""SQLAlchemy models for document storage."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, String, Text
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    doc_id = Column(String(50), unique=True, nullable=False, index=True)
    doc_name = Column(String(255), nullable=False)
    tree_json = Column(Text, nullable=False)
    pdf_path = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    doc_description = Column(Text, nullable=True)
    company = Column(String(255), nullable=True)
    fiscal_year = Column(String(10), nullable=True)
    filing_type = Column(String(50), nullable=True)
