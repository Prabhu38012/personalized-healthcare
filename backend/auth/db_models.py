"""
SQLAlchemy database models for user authentication
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import Column, String, Boolean, DateTime, Integer, Text, Float, JSON
from sqlalchemy.orm import declarative_base
try:
    from backend.db import Base
except ImportError:
    # Fallback for when running from backend directory
    from db import Base

class User(Base):
    """SQLAlchemy User model for persistent storage"""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=False)
    hashed_password = Column(Text, nullable=False)
    role = Column(String, nullable=False, default="patient")
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, nullable=True)
    login_attempts = Column(Integer, default=0, nullable=False)
    locked_until = Column(DateTime, nullable=True)
    
    def __repr__(self):
        return f"<User(id='{self.id}', email='{self.email}', role='{self.role}')>"
