from datetime import datetime
import os
from pathlib import Path

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATABASE_URL = f"sqlite:///{BASE_DIR / 'depression.db'}"
DATABASE_URL = os.environ.get("DATABASE_URL") or DEFAULT_DATABASE_URL

engine_kwargs = {}
if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **engine_kwargs)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()


# PATIENT TABLE


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True)
    patient_id = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    sessions = relationship("Session", back_populates="patient")

class AVInterview(Base):
    __tablename__ = "av_interview"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"))

    question = Column(String)
    transcript = Column(String)

    video_emotion = Column(String)
    audio_emotion = Column(String)
    audio_confidence = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow)

# SESSION TABLE


class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    session_number = Column(Integer)  # e.g., Session 1, 2026-02-12
    session_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="sessions")
    tests = relationship("Test", back_populates="session")



# TEST TABLE


class Test(Base):
    __tablename__ = "tests"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"))

    test_type = Column(String)
    score = Column(Float)
    level = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="tests")


Base.metadata.create_all(bind=engine)
