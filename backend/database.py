from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime

DATABASE_URL = "sqlite:///backend/depression.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

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
