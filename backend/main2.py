from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import Dict
from backend.inference import predict
from backend.utils.preprocess import preprocess_text
from backend.database import SessionLocal, Patient, Session, Test, AVInterview
from backend.av_pipeline import run_av_pipeline          # ← NEW
import torch
from datetime import datetime, timedelta
from fastapi import UploadFile, File, Form
import shutil
app = FastAPI()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SUICIDE_KEYWORDS = [
    "kill myself", "end my life", "suicide",
    "better off dead", "not worth living"
]

def detect_suicide_risk(text):
    text = text.lower()
    return any(word in text for word in SUICIDE_KEYWORDS)

# ─────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# SCORING HELPERS
# ─────────────────────────────────────────────

def clamp_score(score: float):
    return round(max(0, min(score, 10)), 2)


def normalize_score(raw_score: int, max_score: int):
    return clamp_score((raw_score / max_score) * 10)


def interpret_level(score: float):
    if score <= 3:
        return "Low"
    elif score <= 6:
        return "Moderate"
    elif score <= 8:
        return "High"
    else:
        return "Critical"


# ─────────────────────────────────────────────
# WEIGHT CONFIG  (must sum to 1.0)
# ─────────────────────────────────────────────

WEIGHTS = {
    "AI":           0.22,
    "PHQ9":         0.22,
    "AV":           0.10,
    "ANXIETY":      0.13,
    "WORKSTRESS":   0.09,
    "ANGER":        0.09,
    "TRAUMA":       0.05,
    "GRIEF":        0.05,
    "RELATIONSHIP": 0.03,
    "SELF_ESTEEM":  0.01,
    "MENTAL_HEALTH":0.01,
}
# Verify weights sum to 1.0
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, "WEIGHTS must sum to 1.0"


# ─────────────────────────────────────────────
# SESSION HELPERS
# ─────────────────────────────────────────────

def get_or_create_patient(db, user_id):
    patient = db.query(Patient).filter(
        Patient.patient_id == user_id
    ).first()
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
        session_number=last_session_number + 1
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
        level=level
    )
    db.add(test)
    db.commit()

    db.refresh(session)
    session.session_score = calculate_session_score(session)
    db.commit()
    db.close()
    return score, level


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

# AI text assessment
@app.post("/assessments/create")
async def create_assessment(request: AssessmentRequest):
    inputs = preprocess_text(
        request.patient_text,
        request.relative_text,
        DEVICE
    )
    model_score = float(predict(inputs["input_ids"], inputs["attention_mask"]))
    ai_score = clamp_score(model_score)
    score, level = save_test(request.user_id, "AI", ai_score)
    return {"score": score, "level": level}


# PHQ-9
@app.post("/phq9/submit")
def submit_phq9(request: PHQ9Request):
    raw_score = sum(request.answers.values())
    phq_score = normalize_score(raw_score, 27)
    score, level = save_test(request.user_id, "PHQ9", phq_score)
    return {"score": score, "level": level}


# Multi-domain questionnaire
@app.post("/questionnaire/submit")
def submit_questionnaire(request: QuestionnaireRequest):
    max_scores = {
        "anxiety": 21, "self_esteem": 18, "procrastination": 15,
        "workstress": 15, "trauma": 15, "grief": 15,
        "relationship": 15, "anger": 15, "mental_health": 18
    }
    if request.module not in max_scores:
        return {"error": "Invalid module"}
    raw_score = sum(request.answers.values())
    module_score = normalize_score(raw_score, max_scores[request.module])
    score, level = save_test(
        request.user_id, request.module.upper(), module_score
    )
    return {"score": score, "level": level}


# Audio-Video Assessment

@app.post("/av/analyze")
def analyze_av(request: AVRequest):
    """
    Triggers the AV pipeline (recording + analysis) and saves the result.
    This is a blocking call — the client should wait for the full duration.
    """
    result = run_av_pipeline(duration=request.duration)

    score, level = save_test(request.user_id, "AV", result["score"])

    return {
        "score":              score,
        "level":              level,
        "transcript":         result["transcript"],
        "video_emotion":      result["video_emotion"],
        "avg_video_emotions": result["avg_video_emotions"],
        "audio_emotion":      result["audio_emotion"],
        "audio_confidence":   result["audio_confidence"],
        "voice_features":     result["voice_features"],
    }

@app.post("/av/question")
async def analyze_question(
    user_id: str = Form(...),
    question: str = Form(...),
    video: UploadFile = File(...),
    audio: UploadFile = File(...)
):
    """
    Receives video and audio files, analyzes them, and saves interview response to database.
    """
    import os
    os.makedirs("temp", exist_ok=True)
    
    video_path = f"temp/{video.filename}"
    audio_path = f"temp/{audio.filename}"

    # Save uploaded files
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    from backend.av_pipeline import (
        analyze_video_emotion, speech_to_text,
        predict_audio_emotion, extract_audio_from_video,
    )

    try:
        # If the client sent a separate audio file use it directly;
        # otherwise extract audio from the video with PyAV (no ffmpeg CLI).
        if os.path.getsize(audio_path) < 1024:          # empty / stub upload
            extract_audio_from_video(video_path, audio_path)

        # Analyze the files
        video_emotion, avg_video_emotions = analyze_video_emotion(video_path)
        transcript = speech_to_text(audio_path)
        audio_emotion, audio_confidence = predict_audio_emotion(audio_path)
        
        # Save to database
        db = SessionLocal()
        patient = get_or_create_patient(db, user_id)
        session = get_or_create_session(db, patient)
        
        interview = AVInterview(
            session_id=session.id,
            question=question,
            transcript=transcript,
            video_emotion=video_emotion or "neutral",
            audio_emotion=audio_emotion,
            audio_confidence=audio_confidence
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
            "suicide_risk": detect_suicide_risk(transcript)
        }
    except Exception as e:
        return {"error": f"Error analyzing question: {str(e)}"}
    finally:
        # Cleanup temp files
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)

@app.post("/llm/summary")
def llm_summary(data: dict):

    combined = " ".join([r["transcript"] for r in data["responses"]])

    return {
        "summary": f"Patient shows emotional distress. Transcript: {combined[:300]}"
    }
# ─────────────────────────────────────────────
# HISTORY ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/sessions/{user_id}")
def get_sessions(user_id: str):
    db = SessionLocal()
    patient = db.query(Patient).filter(Patient.patient_id == user_id).first()
    if not patient:
        db.close()
        return []
    sessions = (
        db.query(Session)
        .filter(Session.patient_id == patient.id)
        .order_by(Session.created_at.asc())
        .all()
    )
    result = [
        {
            "session_id":     s.id,
            "session_number": s.session_number,
            "session_score":  s.session_score,
            "created_at":     s.created_at,
        }
        for s in sessions
    ]
    db.close()
    return result


@app.get("/tests/{user_id}")
def get_tests(user_id: str):
    db = SessionLocal()
    patient = db.query(Patient).filter(Patient.patient_id == user_id).first()
    if not patient:
        return []
    sessions = db.query(Session).filter(Session.patient_id == patient.id).all()
    results = [
        {
            "session_id": s.id,
            "test_type":  t.test_type,
            "score":      t.score,
            "level":      t.level,
            "created_at": t.created_at,
        }
        for s in sessions
        for t in s.tests
    ]
    db.close()
    return results


@app.get("/interview/{user_id}")
def get_interview_responses(user_id: str):
    """Retrieve all interview responses for a user, organized by session."""
    db = SessionLocal()
    patient = db.query(Patient).filter(Patient.patient_id == user_id).first()
    if not patient:
        db.close()
        return []
    
    sessions = db.query(Session).filter(Session.patient_id == patient.id).all()
    results_by_session = []
    
    for session in sessions:
        interviews = db.query(AVInterview).filter(AVInterview.session_id == session.id).all()
        session_interviews = []
        for interview in interviews:
            session_interviews.append({
                "question": interview.question,
                "transcript": interview.transcript,
                "video_emotion": interview.video_emotion,
                "audio_emotion": interview.audio_emotion,
                "audio_confidence": interview.audio_confidence,
                "created_at": interview.created_at,
            })
        
        if session_interviews:  # Only include sessions with interviews
            results_by_session.append({
                "session_number": session.session_number,
                "session_id": session.id,
                "session_created_at": session.created_at,
                "interviews": session_interviews
            })
    
    db.close()
    return results_by_session