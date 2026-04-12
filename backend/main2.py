from datetime import datetime, timedelta
import os
from pathlib import Path
import shutil
import sys
from typing import Dict

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel
import torch
from sqlalchemy.orm import selectinload

from backend.database import AVInterview, Patient, Session, SessionLocal, Test
from backend.inference import predict_text

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACKEND_HOST = os.environ.get("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.environ.get("BACKEND_PORT", "8000"))

SUICIDE_KEYWORDS = [
    "kill myself", "end my life", "suicide",
    "better off dead", "not worth living",
]


def detect_suicide_risk(text):
    text = text.lower()
    return any(word in text for word in SUICIDE_KEYWORDS)


class AssessmentRequest(BaseModel):
    user_id: str
    patient_text: str
    relative_text: str = ""


class PHQ9Request(BaseModel):
    user_id: str
    answers: Dict[str, int]


class QuestionnaireRequest(BaseModel):
    user_id: str
    module: str
    answers: Dict[str, int]


class AVRequest(BaseModel):
    user_id: str
    duration: int = 60


class MultimodalRequest(BaseModel):
    user_id: str
    patient_text: str
    relative_text: str = ""
    video_path: str
    audio_path: str | None = None


def clamp_score(score: float):
    return round(max(0, min(score, 10)), 2)


def normalize_score(raw_score: int, max_score: int):
    return clamp_score((raw_score / max_score) * 10)


def interpret_level(score: float):
    if score <= 3:
        return "Low"
    if score <= 6:
        return "Moderate"
    if score <= 8:
        return "High"
    return "Critical"


WEIGHTS = {
    "AI": 0.25,
    "PHQ9": 0.25,
    "AV": 0.0,
    "ANXIETY": 0.15,
    "WORKSTRESS": 0.10,
    "ANGER": 0.10,
    "TRAUMA": 0.05,
    "GRIEF": 0.05,
    "RELATIONSHIP": 0.03,
    "SELF_ESTEEM": 0.01,
    "MENTAL_HEALTH": 0.01,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, "WEIGHTS must sum to 1.0"


def get_or_create_patient(db, user_id):
    patient = db.query(Patient).filter(Patient.patient_id == user_id).first()
    if not patient:
        patient = Patient(patient_id=user_id)
        db.add(patient)
        db.commit()
        db.refresh(patient)
    return patient


def get_or_create_session(db, patient):
    now = datetime.utcnow()
    one_hour_ago = now - timedelta(hours=1)

    latest_session = (
        db.query(Session)
        .filter(Session.patient_id == patient.id)
        .order_by(Session.created_at.desc())
        .first()
    )

    if latest_session and latest_session.created_at >= one_hour_ago:
        return latest_session

    last_session_number = (
        db.query(Session)
        .filter(Session.patient_id == patient.id)
        .count()
    )

    new_session = Session(
        patient_id=patient.id,
        session_number=last_session_number + 1,
    )
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return new_session


def calculate_session_score(session):
    total_weighted = 0
    total_weight = 0
    for test in session.tests:
        if test.test_type in WEIGHTS:
            weight = WEIGHTS[test.test_type]
            total_weighted += test.score * weight
            total_weight += weight
    if total_weight == 0:
        return 0
    return clamp_score(total_weighted / total_weight)


def save_test(user_id, test_type, score):
    db = SessionLocal()
    patient = get_or_create_patient(db, user_id)
    session = get_or_create_session(db, patient)
    level = interpret_level(score)

    test = Test(
        session_id=session.id,
        test_type=test_type,
        score=score,
        level=level,
    )
    db.add(test)
    db.commit()

    db.refresh(session)
    session.session_score = calculate_session_score(session)
    db.commit()
    db.close()
    return score, level


@app.post("/assessments/create")
async def create_assessment(request: AssessmentRequest):
    ai_score = clamp_score(
        predict_text(request.patient_text, request.relative_text)
    )
    score, level = save_test(request.user_id, "AI", ai_score)
    return {"score": score, "level": level}


@app.post("/phq9/submit")
def submit_phq9(request: PHQ9Request):
    raw_score = sum(request.answers.values())
    phq_score = normalize_score(raw_score, 27)
    score, level = save_test(request.user_id, "PHQ9", phq_score)
    return {"score": score, "level": level}


@app.post("/questionnaire/submit")
def submit_questionnaire(request: QuestionnaireRequest):
    max_scores = {
        "anxiety": 21,
        "self_esteem": 18,
        "procrastination": 15,
        "workstress": 15,
        "trauma": 15,
        "grief": 15,
        "relationship": 15,
        "anger": 15,
        "mental_health": 18,
    }
    if request.module not in max_scores:
        return {"error": "Invalid module"}
    raw_score = sum(request.answers.values())
    module_score = normalize_score(raw_score, max_scores[request.module])
    score, level = save_test(
        request.user_id,
        request.module.upper(),
        module_score,
    )
    return {"score": score, "level": level}


@app.post("/av/analyze")
def analyze_av(request: AVRequest):
    from backend.av_pipeline import run_av_pipeline

    result = run_av_pipeline(duration=request.duration)
    score, level = save_test(request.user_id, "AV", result["score"])

    return {
        "score": score,
        "level": level,
        "transcript": result["transcript"],
        "video_emotion": result["video_emotion"],
        "avg_video_emotions": result["avg_video_emotions"],
        "audio_emotion": result["audio_emotion"],
        "audio_confidence": result["audio_confidence"],
        "voice_features": result["voice_features"],
    }


@app.post("/multimodal/assess")
def assess_multimodal(request: MultimodalRequest):
    from backend.services.multimodal_service import run_multimodal_inference

    result = run_multimodal_inference(
        patient_text=request.patient_text,
        relative_text=request.relative_text,
        video_path=request.video_path,
        audio_path=request.audio_path,
    )
    score, level = save_test(request.user_id, "AI", result.fused_score)
    return {
        "score": score,
        "level": level,
        "text_score": result.text_score,
        "av_score": result.av_score,
        "fused_score": result.fused_score,
        "transcript": result.transcript,
        "video_emotion": result.video_emotion,
        "avg_video_emotions": result.avg_video_emotions,
        "audio_emotion": result.audio_emotion,
        "audio_confidence": result.audio_confidence,
        "voice_features": result.voice_features,
        "audio_vector": result.audio_vector,
        "video_vector": result.video_vector,
        "model_source": result.model_source,
    }


@app.post("/av/question")
async def analyze_question(
    user_id: str = Form(...),
    question: str = Form(...),
    video: UploadFile = File(...),
    audio: UploadFile = File(...),
):
    TEMP_DIR.mkdir(exist_ok=True)

    video_path = TEMP_DIR / video.filename
    audio_path = TEMP_DIR / audio.filename

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    from backend.av_pipeline import (
        analyze_video_emotion,
        extract_audio_from_video,
        predict_audio_emotion,
        speech_to_text,
    )

    try:
        if audio_path.stat().st_size < 1024:
            extract_audio_from_video(str(video_path), str(audio_path))

        video_emotion, _avg_video_emotions = analyze_video_emotion(str(video_path))
        transcript = speech_to_text(str(audio_path))
        audio_emotion, audio_confidence = predict_audio_emotion(str(audio_path))

        db = SessionLocal()
        patient = get_or_create_patient(db, user_id)
        session = get_or_create_session(db, patient)

        interview = AVInterview(
            session_id=session.id,
            question=question,
            transcript=transcript,
            video_emotion=video_emotion or "neutral",
            audio_emotion=audio_emotion,
            audio_confidence=audio_confidence,
        )
        db.add(interview)
        db.commit()
        db.close()

        return {
            "question": question,
            "transcript": transcript,
            "video_emotion": video_emotion or "neutral",
            "audio_emotion": audio_emotion,
            "audio_confidence": audio_confidence,
            "suicide_risk": detect_suicide_risk(transcript),
        }
    except Exception as e:
        return {"error": f"Error analyzing question: {str(e)}"}
    finally:
        if video_path.exists():
            video_path.unlink()
        if audio_path.exists():
            audio_path.unlink()


@app.post("/llm/summary")
def llm_summary(data: dict):
    combined = " ".join([r["transcript"] for r in data["responses"]])
    return {
        "summary": f"Patient shows emotional distress. Transcript: {combined[:300]}",
    }


@app.get("/sessions/{user_id}")
def get_sessions(user_id: str):
    db = SessionLocal()
    try:
        patient = db.query(Patient).filter(Patient.patient_id == user_id).first()
        if not patient:
            return []
        sessions = (
            db.query(Session)
            .options(selectinload(Session.tests))
            .filter(Session.patient_id == patient.id)
            .order_by(Session.created_at.asc())
            .all()
        )
        return [
            {
                "session_id": s.id,
                "session_number": s.session_number,
                "session_score": s.session_score,
                "created_at": s.created_at,
            }
            for s in sessions
        ]
    finally:
        db.close()


@app.get("/tests/{user_id}")
def get_tests(user_id: str):
    db = SessionLocal()
    try:
        patient = db.query(Patient).filter(Patient.patient_id == user_id).first()
        if not patient:
            return []
        sessions = (
            db.query(Session)
            .options(selectinload(Session.tests))
            .filter(Session.patient_id == patient.id)
            .order_by(Session.created_at.asc())
            .all()
        )
        results = [
            {
                "session_id": s.id,
                "test_type": t.test_type,
                "score": t.score,
                "level": t.level,
                "created_at": t.created_at,
            }
            for s in sessions
            for t in s.tests
        ]
        return results
    finally:
        db.close()


@app.get("/interview/{user_id}")
def get_interview_responses(user_id: str):
    db = SessionLocal()
    try:
        patient = db.query(Patient).filter(Patient.patient_id == user_id).first()
        if not patient:
            return []

        sessions = (
            db.query(Session)
            .filter(Session.patient_id == patient.id)
            .order_by(Session.created_at.asc())
            .all()
        )
        results_by_session = []

        for session in sessions:
            interviews = (
                db.query(AVInterview)
                .filter(AVInterview.session_id == session.id)
                .order_by(AVInterview.created_at.asc())
                .all()
            )
            session_interviews = [
                {
                    "question": interview.question,
                    "transcript": interview.transcript,
                    "video_emotion": interview.video_emotion,
                    "audio_emotion": interview.audio_emotion,
                    "audio_confidence": interview.audio_confidence,
                    "created_at": interview.created_at,
                }
                for interview in interviews
            ]

            if session_interviews:
                results_by_session.append(
                    {
                        "session_number": session.session_number,
                        "session_id": session.id,
                        "session_created_at": session.created_at,
                        "interviews": session_interviews,
                    }
                )

        return results_by_session
    finally:
        db.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main2:app", host=BACKEND_HOST, port=BACKEND_PORT, reload=False)
