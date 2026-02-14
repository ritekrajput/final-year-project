from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Dict
from backend.inference import predict
from backend.utils.preprocess import preprocess_text
from backend.database import SessionLocal, Patient, Session, Test
import torch
from datetime import datetime, timedelta

app = FastAPI()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================
# ----------------------- MODELS -----------------------
# ======================================================

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


# ======================================================
# ---------------- STANDARD SCORING -------------------
# ======================================================

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


# ======================================================
# ---------------- WEIGHT CONFIG -----------------------
# ======================================================

WEIGHTS = {
    "AI": 0.25,
    "PHQ9": 0.25,
    "ANXIETY": 0.15,
    "WORKSTRESS": 0.10,
    "ANGER": 0.10,
    "TRAUMA": 0.05,
    "GRIEF": 0.05,
    "RELATIONSHIP": 0.03,
    "SELF_ESTEEM": 0.01,
    "MENTAL_HEALTH": 0.01
}


# ======================================================
# ---------------- SESSION HELPERS --------------------
# ======================================================

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

    # If within 1 hour → same session
    if latest_session and latest_session.created_at >= one_hour_ago:
        return latest_session

    # Otherwise create new session
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


# ======================================================
# ---------------- AI ENDPOINT ------------------------
# ======================================================

@app.post("/assessments/create")
async def create_assessment(request: AssessmentRequest):

    inputs = preprocess_text(
        request.patient_text,
        request.relative_text,
        DEVICE
    )

    model_score = float(
        predict(inputs["input_ids"], inputs["attention_mask"])
    )

    ai_score = clamp_score(model_score)

    score, level = save_test(
        request.user_id,
        "AI",
        ai_score
    )

    return {
        "score": score,
        "level": level
    }


# ======================================================
# ---------------- PHQ-9 ENDPOINT ---------------------
# ======================================================

@app.post("/phq9/submit")
def submit_phq9(request: PHQ9Request):

    raw_score = sum(request.answers.values())
    phq_score = normalize_score(raw_score, 27)

    score, level = save_test(
        request.user_id,
        "PHQ9",
        phq_score
    )

    return {
        "score": score,
        "level": level
    }


# ======================================================
# -------- MULTI-DOMAIN QUESTIONNAIRE ENDPOINT -------
# ======================================================

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
        "mental_health": 18
    }

    if request.module not in max_scores:
        return {"error": "Invalid module"}

    raw_score = sum(request.answers.values())
    module_score = normalize_score(raw_score, max_scores[request.module])

    score, level = save_test(
        request.user_id,
        request.module.upper(),
        module_score
    )

    return {
        "score": score,
        "level": level
    }


# ======================================================
# ---------------- SESSION VIEW ENDPOINT --------------
# ======================================================

@app.get("/sessions/{user_id}")
def get_sessions(user_id: str):

    db = SessionLocal()

    patient = db.query(Patient).filter(
        Patient.patient_id == user_id
    ).first()

    if not patient:
        db.close()
        return []

    sessions = db.query(Session).filter(
        Session.patient_id == patient.id
    ).order_by(Session.created_at.asc()).all()

    result = []

    for s in sessions:
        result.append({
            "session_id": s.id,
            "session_number": s.session_number,   
            "session_score": s.session_score,
            "created_at": s.created_at
        })

    db.close()

    return result



# ======================================================
# ---------------- TEST HISTORY ENDPOINT --------------
# ======================================================

@app.get("/tests/{user_id}")
def get_tests(user_id: str):

    db = SessionLocal()

    patient = db.query(Patient).filter(
        Patient.patient_id == user_id
    ).first()

    if not patient:
        return []

    sessions = db.query(Session).filter(
        Session.patient_id == patient.id
    ).all()

    results = []

    for s in sessions:
        for t in s.tests:
            results.append({
                "session_id": s.id,
                "test_type": t.test_type,
                "score": t.score,
                "level": t.level,
                "created_at": t.created_at
            })

    db.close()

    return results
