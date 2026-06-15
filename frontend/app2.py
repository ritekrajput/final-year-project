import streamlit as st
import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from contextlib import suppress
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import time
import threading
import cv2

# ─────────────────────────────────────────────
# AUDIO + VIDEO RECORDER
# Records mic audio via sounddevice in a background thread (no ffmpeg needed)
# ─────────────────────────────────────────────

AUDIO_SR = 16000
MAX_RECORD_SECONDS = float(os.environ.get("MAX_RECORD_SECONDS", "30"))

class VideoRecorder(VideoProcessorBase):
    def __init__(self):
        self.frames = []
        self.audio_chunks = []        # raw int16 PCM chunks from sounddevice
        self.recording = False
        self.start_time = None
        self.lock = threading.Lock()
        self._audio_thread = None
        self._stop_audio = threading.Event()

    # ── audio capture (runs in its own thread) ──────────────────────────────
    def _record_audio(self):
        import sounddevice as sd
        import numpy as np

        def callback(indata, frames, time_info, status):
            if self.recording:
                with self.lock:
                    self.audio_chunks.append(indata.copy())

        with sd.InputStream(
            samplerate=AUDIO_SR,
            channels=1,
            dtype="int16",
            callback=callback,
            blocksize=1024,
        ):
            self._stop_audio.wait()   # block until stop_recording() is called

    # ── public API ──────────────────────────────────────────────────────────
    def start_recording(self):
        with self.lock:
            self.frames = []
            self.audio_chunks = []
            self.recording = True
            self.start_time = time.time()

        self._stop_audio.clear()
        self._audio_thread = threading.Thread(
            target=self._record_audio, daemon=True
        )
        self._audio_thread.start()

    def stop_recording(self):
        self.recording = False
        self._stop_audio.set()
        if self._audio_thread:
            self._audio_thread.join(timeout=3)

    def get_audio_array(self):
        """Return captured audio as a 1-D int16 numpy array at AUDIO_SR."""
        import numpy as np
        with self.lock:
            if not self.audio_chunks:
                return np.zeros(AUDIO_SR, dtype="int16")   # 1 s silence fallback
            return np.concatenate(self.audio_chunks).flatten()

    # ── webrtc frame handler ─────────────────────────────────────────────────
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        with self.lock:
            if self.recording and self.start_time is not None:
                elapsed = time.time() - self.start_time
                if elapsed > MAX_RECORD_SECONDS:
                    self.recording = False
                    self._stop_audio.set()
                else:
                    self.frames.append(img.copy())
        return frame


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

BASE_URL = os.environ.get("APP_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
TEMP_DIR = Path("temp")
DEFAULT_API_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "30"))
DEFAULT_AV_TIMEOUT = float(os.environ.get("AV_REQUEST_TIMEOUT", "180"))
DEFAULT_HISTORY_TIMEOUT = float(os.environ.get("HISTORY_REQUEST_TIMEOUT", "90"))

st.set_page_config(page_title="Mental Health Assessment", layout="centered")


def render_landing_page():
    st.markdown(
        """
        <style>
            .landing-shell {
                max-width: 860px;
                margin: 0 auto;
                padding: 1rem 0 2rem 0;
            }
            .hero-card {
                background: #121826;
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 20px;
                padding: 1.8rem 2rem;
                box-shadow: 0 14px 32px rgba(0, 0, 0, 0.28);
            }
            .hero-badge {
                display: inline-block;
                padding: 0.3rem 0.72rem;
                border-radius: 999px;
                background: rgba(34, 197, 94, 0.16);
                color: #dcfce7;
                font-size: 0.82rem;
                font-weight: 700;
                margin-bottom: 0.9rem;
            }
            .hero-card h1,
            .section-card h3,
            .hero-card p,
            .section-card li,
            .landing-footer {
                color: #ffffff;
            }
            .hero-card p {
                color: #d1d5db;
                line-height: 1.6;
                font-size: 1rem;
                margin-bottom: 0;
            }
            .section-card {
                background: #0f172a;
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 18px;
                padding: 1.05rem 1.2rem;
                margin-top: 1rem;
                box-shadow: 0 12px 28px rgba(0, 0, 0, 0.22);
            }
            .section-card h3 {
                margin-bottom: 0.6rem;
                font-size: 1.05rem;
            }
            .section-card ul {
                margin-bottom: 0;
                padding-left: 1.25rem;
            }
            .section-card li {
                margin-bottom: 0.45rem;
                line-height: 1.5;
                color: #d1d5db;
            }
            .disclaimer-box {
                border-left: 5px solid #38bdf8;
            }
            .instructions-box {
                border-left: 5px solid #34d399;
            }
            .components-box {
                border-left: 5px solid #a78bfa;
            }
            .checklist {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 0.6rem;
                margin-top: 0.85rem;
            }
            .check-item {
                background: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 12px;
                padding: 0.7rem 0.8rem;
                color: #e5e7eb;
                font-size: 0.9rem;
            }
            .landing-footer {
                color: #cbd5e1;
                font-size: 0.92rem;
                margin-top: 0.8rem;
                text-align: center;
            }
            div[data-testid="stCheckbox"] label {
                color: #e5e7eb;
                font-weight: 500;
            }
            div[data-testid="stButton"] button {
                height: 3rem;
                border-radius: 14px;
                font-weight: 700;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="landing-shell">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-badge">Mental health screening portal</div>
            <h1 style="margin-bottom: 0.6rem;">Mental Health Assessment Portal</h1>
            <p>
                Please review the disclaimer and instructions below before proceeding to the assessment workflow.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-card disclaimer-box">
            <h3>Disclaimer</h3>
            <ul>
                <li>This assessment tool is intended for educational and screening purposes only.</li>
                <li>The patient should not be under influence of alcohol or any drug during the assesment. </li>
                <li>It is not a substitute for professional medical diagnosis or treatment.</li>
                <li>Results should be interpreted as supportive insights and not clinical conclusions.</li>
                <li>If you are experiencing severe distress or thoughts of self-harm, please seek immediate professional help.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-card instructions-box">
            <h3>Instructions</h3>
            <ul>
                <li>Complete all assessments honestly and thoughtfully.</li>
                <li>The platform may include text-based, questionnaire-based, audio-based, and video-based assessments.</li>
                <li>Some sections may ask open-ended questions about emotions, behavior, mood, and daily experiences.</li>
                <li>Audio and video responses should be recorded in a quiet environment with adequate lighting.</li>
                <li>There are no right or wrong answers.</li>
                <li>Responses are used only for generating a mental health severity assessment score.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-card components-box">
            <h3>Assessment Components</h3>
            <div class="checklist">
                <div class="check-item">AI Free-Text Assessment</div>
                <div class="check-item">PHQ-9 Questionnaire</div>
                <div class="check-item">Multi-Domain Mental Health Screening</div>
                <div class="check-item">Audio-Based Assessment</div>
                <div class="check-item">Video-Based Assessment</div>
                <div class="check-item">Relative/Observer Assessment (if applicable)</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='landing-footer'>By proceeding, you acknowledge that you have read and understood the information above.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
    consent = st.checkbox(
        "I have read and understood the disclaimer and instructions.",
        key="assessment_gate_consent",
    )

    proceed = st.button(
        "Proceed to Assessment",
        type="primary",
        use_container_width=True,
        disabled=not consent,
    )
    if proceed:
        st.session_state["assessment_gate_accepted"] = True
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def api_request(method, path, *, context, timeout=None, **kwargs):
    request_timeout = DEFAULT_API_TIMEOUT if timeout is None else timeout
    url = f"{BASE_URL}{path}"

    try:
        response = requests.request(method, url, timeout=request_timeout, **kwargs)
    except requests.Timeout:
        st.error(f"{context} timed out after {request_timeout:.0f}s.")
        return None
    except requests.RequestException as exc:
        st.error(f"{context} failed: {exc}")
        return None

    if not response.ok:
        details = response.text.strip()
        if len(details) > 400:
            details = details[:400] + "..."
        st.error(f"{context} failed with HTTP {response.status_code}: {details or 'No response body'}")
        return None

    try:
        return response.json()
    except ValueError:
        st.error(f"{context} returned an invalid JSON response.")
        return None


def set_flash(message, level="success"):
    st.session_state["flash_message"] = message
    st.session_state["flash_level"] = level


def show_flash():
    message = st.session_state.get("flash_message")
    if not message:
        return

    level = st.session_state.get("flash_level", "info")
    if level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    elif level == "error":
        st.error(message)
    else:
        st.info(message)

    st.session_state.pop("flash_message", None)
    st.session_state.pop("flash_level", None)


def cleanup_paths(*paths):
    for path in paths:
        if not path:
            continue
        with suppress(Exception):
            Path(path).unlink(missing_ok=True)


def compute_guided_interview_score(responses):
    if not responses:
        return 0.0, "Low"

    emotion_scores = {
        "calm": 1.0,
        "neutral": 2.0,
        "stable": 1.2,
        "slightly sad": 4.0,
        "tired": 4.5,
        "low energy": 4.5,
        "flat": 4.2,
        "reserved": 4.0,
        "uneasy": 5.5,
        "worried": 6.0,
        "anxious": 7.0,
        "stressed": 7.2,
        "distressed": 8.2,
        "overwhelmed": 8.8,
        "guarded": 5.0,
    }

    total = 0.0
    count = 0
    for response in responses:
        audio_emotion = str(response.get("audio_emotion", "")).strip().lower()
        video_emotion = str(response.get("video_emotion", "")).strip().lower()
        audio_confidence = float(response.get("audio_confidence", 0) or 0)
        suicide_risk = bool(response.get("suicide_risk", False))

        audio_score = emotion_scores.get(audio_emotion, 3.0)
        video_score = emotion_scores.get(video_emotion, 3.0)
        confidence_adjustment = (1.0 - min(max(audio_confidence, 0.0), 1.0)) * 1.5
        risk_boost = 2.0 if suicide_risk else 0.0

        total += ((audio_score + video_score) / 2.0) + confidence_adjustment + risk_boost
        count += 1

    score = round(min(max(total / count, 0.0), 10.0), 2)
    if score <= 3:
        level = "Low"
    elif score <= 6:
        level = "Moderate"
    elif score <= 8:
        level = "High"
    else:
        level = "Critical"

    return score, level

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

if not st.session_state.get("assessment_gate_accepted", False):
    render_landing_page()
    st.stop()

page = st.sidebar.radio(
    "Navigation",
    [
        "PHQ-9 Questionnaire",
        "Mood Disorder Assessment",
        "Audio/Video Assessment",
        "Patient History Dashboard",
    ]
)


# ─────────────────────────────────────────────
# PAGE 1 — AI TEXT ASSESSMENT
# ─────────────────────────────────────────────

if page == "Run AI Assessment":

    st.title("AI Depression Assessment")

    user_id = st.text_input("Patient ID")
    patient_text = st.text_area("How have you been feeling recently?")
    relative_text = st.text_area("Observed behavioral changes")

    if st.button("Run AI Assessment"):
        if not user_id or not patient_text:
            st.error("Patient ID and text input required.")
        else:
            result = api_request(
                "post",
                "/assessments/create",
                context="AI assessment",
                json={
                    "user_id": user_id,
                    "patient_text": patient_text,
                    "relative_text": relative_text,
                },
            )
            if result:
                st.success("Assessment Complete")
                st.metric("AI Score", f"{result['score']} / 10")
                st.metric("Risk Level", result["level"])


# ─────────────────────────────────────────────
# PAGE 2 — PHQ-9
# ─────────────────────────────────────────────

elif page == "PHQ-9 Questionnaire":

    st.title("PHQ-9 Questionnaire")

    user_id = st.text_input("Patient ID")

    questions = [
            "Over the past two weeks, have you found it hard to enjoy things you normally like doing?",
    
    "Over the past two weeks, have you often felt sad, low, or hopeless?",
    
    "Over the past two weeks, have you been having trouble sleeping, or sleeping more than usual?",
    
    "Over the past two weeks, have you felt tired or low on energy most of the time?",
    
    "Over the past two weeks, have you noticed changes in your eating habits or appetite?",
    
    "Over the past two weeks, have you been hard on yourself or felt like you're not doing well enough?",
    
    "Over the past two weeks, have you found it difficult to concentrate on studies, work, reading, or everyday activities?",
    
    "Over the past two weeks, have people noticed that you seem unusually slow, or have you felt unusually restless or unable to relax?",
    
    "Over the past two weeks, have you had thoughts that you would be better off not being here, or thoughts of hurting yourself?"
    ]

    options = [   "Not at all (0)",
    "Sometimes (1)",
    "More frequently (2)",
    "Almost every day (3)"
]

    answers = {}
    for i, q in enumerate(questions):
        choice = st.radio(q, options, key=f"phq_{i}")
        answers[f"q{i+1}"] = options.index(choice)

    if st.button("Submit PHQ-9"):
        if not user_id:
            st.error("Patient ID required.")
        else:
            result = api_request(
                "post",
                "/phq9/submit",
                context="PHQ-9 submission",
                json={"user_id": user_id, "answers": answers},
            )
            if result:
                st.success("Submitted")
                st.metric("PHQ-9 Score", f"{result['score']} / 10")
                st.metric("Risk Level", result["level"])


# ─────────────────────────────────────────────
# PAGE 3 — MULTI-DOMAIN
# ─────────────────────────────────────────────

elif page == "Mood Disorder Assessment":

    st.title("Multi-Domain Mood Disorder Assessment")

    user_id = st.text_input("Patient ID")

    module = st.selectbox(
        "Select Module",
        ["anxiety", "self_esteem", "procrastination",
         "workstress", "trauma", "grief",
         "relationship", "anger", "mental_health"]
    )

    question_bank = {
        "anxiety": [
            "Have you been feeling nervous, anxious, or on edge lately?",
"Have you found it difficult to stop or control your worrying?",
"Have you been worrying too much about different things?",
"Have you been finding it difficult to relax and unwind?",
"Have you felt so restless that it was hard to sit still or stay calm?",
"Have you been getting irritated or annoyed more easily than usual?",
"Have you often felt that something bad might happen, even without a clear reason?",
        ],
        "self_esteem": [
            "Do you generally feel confident in your abilities and decisions?", "Do you feel that you are a valuable and worthwhile person?", "Do you often feel inadequate or not good enough?", "Do you find yourself comparing yourself negatively to other people?", "Do you feel proud of your achievements and accomplishments?", "Do you tend to doubt yourself even when you have succeeded?",],
        "procrastination": [
            "Do you often delay starting important tasks or responsibilities?", "Do you tend to leave assignments, work, or deadlines until the last minute?", "Do you sometimes feel overwhelmed and avoid responsibilities because of it?", "Do you find yourself getting distracted when you should be focusing on important work?", "Do you often regret delaying tasks once deadlines get closer?", ],
        
        "workstress": [
            "Have you been feeling overwhelmed by your workload or responsibilities?", "Do deadlines often make you feel stressed or pressured?", "Have you been feeling emotionally exhausted because of work or studies?", "Do you feel significant pressure to perform well academically or professionally?", "Do you worry about your future career, job security, or academic success?",
        ],
        "trauma": [
           "Have you experienced an event that felt deeply distressing or traumatic?", "Do unwanted memories of that event return to your mind frequently?", "Do you try to avoid people, places, or situations that remind you of the event?", "Have you felt emotionally numb or disconnected since the event occurred?", "Do you often feel constantly alert, tense, or easily startled?",
        ],
        "grief": [
            "Have you been experiencing intense sadness because of a loss in your life?", "Do you find it difficult to accept or come to terms with that loss?", "Have you felt that life has lost some of its meaning since the loss occurred?", "Do you avoid situations, places, or memories that remind you of the loss?", "Do you often experience feelings of guilt related to the loss?",
        ],
        "relationship": [
            "Do you feel emotionally supported by the important people in your life?", "Do you often feel misunderstood by your partner, family, or close friends?", "Do you find it difficult to communicate openly with people close to you?", "Do you frequently feel anxious or insecure about your relationships?", "Do you feel that conflicts in your relationships remain unresolved for long periods?",
        ],
        "anger": [ "Do you lose your temper more easily than you would like?", "Do you sometimes feel intense anger over situations that later seem minor?", "Do you often regret things you say or do when you are angry?", "Do you find it difficult to calm down once you become upset?", "Do you feel that your anger negatively affects your relationships, studies, work, or daily life?", ],
        
        "mental_health": [ "Do you generally feel emotionally stable and balanced?", "Do you feel motivated to carry out your daily responsibilities?", "Do you feel hopeful and optimistic about your future?", "Have you been finding it difficult to enjoy activities that you usually like?", "Do you often feel emotionally overwhelmed by your thoughts or feelings?", "Do you feel socially connected and supported by people around you?", ],
    }

    reverse_items = {
        "self_esteem":  [0, 1],
        "mental_health":[0, 1, 2, 5],
        "relationship": [0],
    }

    options = ["Not at all (0)",
    "Occasionally (1)",
    "Frequently (2)",
    "Almost every day (3)"]

    answers = {}
    for i, q in enumerate(question_bank[module]):
        choice = st.radio(q, options, key=f"{module}_{i}")
        answers[f"q{i+1}"] = options.index(choice)

    if st.button("Submit Module"):
        if not user_id:
            st.error("Patient ID required.")
        else:
            if module in reverse_items:
                for idx in reverse_items[module]:
                    key = f"q{idx+1}"
                    answers[key] = 3 - answers[key]

            result = api_request(
                "post",
                "/questionnaire/submit",
                context=f"{module.upper()} questionnaire submission",
                json={"user_id": user_id, "module": module, "answers": answers},
            )
            if result:
                st.success("Submitted")
                st.metric(f"{module.upper()} Score", f"{result['score']} / 10")
                st.metric("Risk Level", result["level"])


# ─────────────────────────────────────────────
# PAGE 4 — AUDIO / VIDEO ASSESSMENT
# ─────────────────────────────────────────────

elif page == "Fused Multimodal Assessment":

    st.title("Fused Multimodal Assessment")
    st.caption("Run the paper-style fused text + audio + video pipeline using a recorded interview file.")
    show_flash()

    user_id = st.text_input("Patient ID", key="mm_user_id")
    patient_text = st.text_area("Patient Narrative", key="mm_patient_text")
    relative_text = st.text_area("Relative / Observer Notes", key="mm_relative_text")
    video_file = st.file_uploader(
        "Upload Interview Video",
        type=["mp4", "avi", "mov", "mkv"],
        key="mm_video_file",
    )
    audio_file = st.file_uploader(
        "Optional Separate Audio File",
        type=["wav", "mp3", "m4a"],
        key="mm_audio_file",
    )

    if st.button("Run Fused Multimodal Assessment"):
        if not user_id or not patient_text or video_file is None:
            st.error("Patient ID, patient narrative, and a video file are required.")
        else:
            TEMP_DIR.mkdir(exist_ok=True)
            video_path = TEMP_DIR / f"multimodal_{video_file.name}"
            video_path.write_bytes(video_file.getbuffer())

            audio_path = None
            if audio_file is not None:
                audio_path = TEMP_DIR / f"multimodal_{audio_file.name}"
                audio_path.write_bytes(audio_file.getbuffer())

            try:
                result = api_request(
                    "post",
                    "/multimodal/assess",
                    context="Fused multimodal assessment",
                    timeout=DEFAULT_AV_TIMEOUT,
                    json={
                        "user_id": user_id,
                        "patient_text": patient_text,
                        "relative_text": relative_text,
                        "video_path": str(video_path.resolve()),
                        "audio_path": str(audio_path.resolve()) if audio_path else None,
                    },
                )
                if result:
                    st.success("Multimodal assessment complete")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Fused Score", f"{result['fused_score']} / 10")
                    col2.metric("Text Score", f"{result['text_score']} / 10")
                    col3.metric("AV Score", f"{result['av_score']} / 10")
                    st.metric("Risk Level", result["level"])
                    st.write(f"Model Source: `{result['model_source']}`")
                    st.write(f"Transcript: {result['transcript']}")
                    st.write(f"Video Emotion: {result['video_emotion']}")
                    st.write(
                        f"Audio Emotion: {result['audio_emotion']} "
                        f"(confidence {result['audio_confidence']})"
                    )
                    st.subheader("Voice Features")
                    st.json(result["voice_features"])
                    st.subheader("Video Feature Vector")
                    st.json(result["video_vector"])
            finally:
                cleanup_paths(video_path, audio_path)

elif page == "Audio/Video Assessment":

    st.title("Guided Interview Assessment")
    show_flash()

    user_id = st.text_input("Patient ID")

    questions = [
        "How have you been feeling most days recently?",
        "How easy or difficult is it to get through your normal day?",
        "What’s something you usually enjoy—how does it feel now?",
        "How would you describe your energy levels lately?",
        "How has your sleep been recently?",
        "How clear or focused does your mind feel?",
        "What thoughts do you have about yourself these days?",
        "How do you see your future?",
        "Have you had thoughts of harming yourself?"
    ]

    if "current_q" not in st.session_state:
        st.session_state.current_q = 0
        st.session_state.responses = []
        st.session_state.saved_video = None
        st.session_state.saved_audio = None
        st.session_state.recording_started = False
        st.session_state.recording_stopped = False
        st.session_state.submission_in_progress = False

    if user_id:

        q_idx = st.session_state.current_q

        if q_idx < len(questions):

            st.subheader(f"Question {q_idx+1}")
            st.write(questions[q_idx])
            st.caption(f"Recording limit: {MAX_RECORD_SECONDS:.0f} seconds per answer.")

            from streamlit_webrtc import webrtc_streamer

            # Helper: write frames → AVI, write captured audio → WAV (no ffmpeg)
            def save_and_process_recording(frames, audio_array=None,
                                           video_filename="temp_video.avi",
                                           audio_filename="temp_audio.wav"):
                if not frames:
                    return None, None
                try:
                    import os, numpy as np
                    from scipy.io.wavfile import write as write_wav

                    os.makedirs("temp", exist_ok=True)
                    video_path = os.path.abspath(os.path.join("temp", video_filename))
                    audio_path = os.path.abspath(os.path.join("temp", audio_filename))

                    # ── Save video ───────────────────────────────────────────
                    h, w, _ = frames[0].shape
                    out = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"XVID"),
                        20, (w, h)
                    )
                    for frame in frames:
                        out.write(frame)
                    out.release()

                    if not os.path.exists(video_path):
                        st.error(f"❌ Video not saved: {video_path}")
                        return None, None
                    st.info(f"✓ Video saved: {len(frames)} frames")

                    # ── Save audio (sounddevice captured, no ffmpeg needed) ──
                    if audio_array is not None and len(audio_array) > 0:
                        pcm = audio_array.flatten().astype(np.int16)
                    else:
                        # Genuine silence fallback (should rarely happen)
                        pcm = np.zeros(AUDIO_SR, dtype=np.int16)

                    write_wav(audio_path, AUDIO_SR, pcm)
                    st.info(f"✓ Audio saved: {len(pcm)/AUDIO_SR:.1f}s @ {AUDIO_SR}Hz")
                    return video_path, audio_path

                except Exception as e:
                    st.error(f"❌ Error saving recording: {e}")
                    return None, None

            ctx = webrtc_streamer(
            key=f"q_{q_idx}",
            video_processor_factory=VideoRecorder,
            media_stream_constraints={"video": True, "audio": True},
            async_processing=True,
        )

            if ctx.video_processor:
                if st.session_state.submission_in_progress:
                    st.info("Submitting recording and running analysis...")
                elif not st.session_state.recording_started and not st.session_state.recording_stopped:
                    if st.button("🎥 Start Recording"):
                        ctx.video_processor.start_recording()
                        st.session_state.recording_started = True
                        st.session_state.recording_stopped = False
                        st.session_state.submission_in_progress = False
                        st.session_state.saved_video = None
                        st.session_state.saved_audio = None
                        set_flash("Recording started.", "info")
                        st.rerun()
                elif st.session_state.recording_started:
                    st.info("Recording in progress. Press Stop when you are done.")
                    if st.button("🛑 Stop Recording"):
                        ctx.video_processor.stop_recording()
                        st.session_state.recording_started = False

                        frames = ctx.video_processor.frames
                        audio_array = ctx.video_processor.get_audio_array()

                        if len(frames) > 0:
                            video_file, audio_file = save_and_process_recording(
                                frames, audio_array=audio_array
                            )
                            if video_file and audio_file:
                                st.session_state.saved_video = video_file
                                st.session_state.saved_audio = audio_file
                                st.session_state.recording_stopped = True
                                set_flash(
                                    f"Recording stopped and saved successfully ({len(frames)} frames captured).",
                                    "success",
                                )
                                st.rerun()
                            else:
                                set_flash("Failed to save the recording.", "error")
                                st.rerun()
                        else:
                            set_flash("No frames were captured. Please try recording again.", "warning")
                            st.rerun()
                elif st.session_state.recording_stopped and st.session_state.saved_video and st.session_state.saved_audio:
                    if st.button("✅ Submit Answer"):
                        st.session_state.submission_in_progress = True
                        st.rerun()

            if st.session_state.submission_in_progress:
                if st.session_state.saved_video and st.session_state.saved_audio:
                    try:
                        # Send files to backend for processing and database storage
                        with open(st.session_state.saved_video, "rb") as video_f, \
                             open(st.session_state.saved_audio, "rb") as audio_f:
                            with st.spinner("Uploading recording and running analysis..."):
                                result = api_request(
                                    "post",
                                    "/av/question",
                                    context="AV question submission",
                                    timeout=DEFAULT_AV_TIMEOUT,
                                    files={
                                        "video": ("video.avi", video_f, "video/x-msvideo"),
                                        "audio": ("audio.wav", audio_f, "audio/wav"),
                                    },
                                    data={
                                        "user_id": user_id,
                                        "question": questions[q_idx],
                                    },
                                )

                        if result:
                            st.session_state.responses.append(result)
                            st.session_state.current_q += 1
                            st.session_state.recording_started = False
                            st.session_state.recording_stopped = False
                            st.session_state.submission_in_progress = False
                            cleanup_paths(
                                st.session_state.saved_video,
                                st.session_state.saved_audio,
                            )
                            st.session_state.saved_video = None
                            st.session_state.saved_audio = None
                            set_flash("Answer submitted and saved to the database.", "success")
                            st.rerun()
                    except Exception as e:
                        st.session_state.submission_in_progress = False
                        st.error(f"Error submitting answer: {str(e)}")
                else:
                    st.session_state.submission_in_progress = False
                    st.error("No saved recording found. Please record and stop first.")

        else:
            st.success("✅ Interview Completed!")
            guided_score, guided_level = compute_guided_interview_score(
                st.session_state.responses
            )
            col1, col2 = st.columns(2)
            col1.metric("Interview Score", f"{guided_score} / 10")
            col2.metric("Risk Level", guided_level)

            if st.button("Generate Clinical Summary"):
                summary = api_request(
                    "post",
                    "/llm/summary",
                    context="Clinical summary generation",
                    json={"responses": st.session_state.responses},
                )
                if summary:
                    st.write(summary["summary"])

# ─────────────────────────────────────────────
# PAGE 5 — PATIENT HISTORY DASHBOARD
# ─────────────────────────────────────────────

elif page == "Patient History Dashboard":

    st.title("📊 Patient Dashboard")

    user_id = st.text_input("Enter Patient ID")

    if not user_id:
        st.info("Please enter a Patient ID.")
        st.stop()

    # Fetch data
    tests_data = api_request(
        "get",
        f"/tests/{user_id}",
        context="Loading test history",
        timeout=DEFAULT_HISTORY_TIMEOUT,
    )
    sessions_data = api_request(
        "get",
        f"/sessions/{user_id}",
        context="Loading session history",
        timeout=DEFAULT_HISTORY_TIMEOUT,
    )
    if tests_data is None or sessions_data is None:
        st.stop()

    if not tests_data:
        st.warning("No test records found.")
        st.stop()

    tests_df = pd.DataFrame(tests_data)
    tests_df["created_at"] = pd.to_datetime(tests_df["created_at"])
    tests_df = tests_df.sort_values("created_at")
    dashboard_tests_df = tests_df[tests_df["test_type"] != "AI"].copy()

    sessions_df = pd.DataFrame(sessions_data)
    if not sessions_df.empty:
        sessions_df["created_at"] = pd.to_datetime(sessions_df["created_at"])
        sessions_df = sessions_df.sort_values("created_at")

    # Session selector
    if not sessions_df.empty:
        session_options = sessions_df["session_number"].tolist()
        selected_session = st.selectbox("Select Session", session_options)

        selected_session_data = sessions_df[
            sessions_df["session_number"] == selected_session
        ].iloc[0]

        st.metric(
            "Selected Session Score",
            f"{selected_session_data['session_score']} / 10"
        )

        session_id   = selected_session_data["session_id"]
        session_tests = dashboard_tests_df[dashboard_tests_df["session_id"] == session_id]

        st.subheader("Tests in Selected Session")
        st.dataframe(session_tests)

        st.markdown("---")
        st.subheader("Compare Sessions")

        compare_sessions = st.multiselect(
            "Select Two Sessions to Compare",
            sessions_df["session_number"].tolist()
        )

        if len(compare_sessions) == 2:
            s1 = sessions_df[sessions_df["session_number"] == compare_sessions[0]].iloc[0]
            s2 = sessions_df[sessions_df["session_number"] == compare_sessions[1]].iloc[0]
            diff = round(s2["session_score"] - s1["session_score"], 2)
            st.metric("Session Score Difference", diff)
            if diff > 0:
                st.warning("Risk increased.")
            elif diff < 0:
                st.success("Improvement detected.")
            else:
                st.info("No change.")

    # Sessions overview
    st.subheader("Sessions Overview")

    if not sessions_df.empty:
        st.dataframe(sessions_df)

        fig_session, ax_session = plt.subplots()
        ax_session.plot(
            sessions_df["created_at"],
            sessions_df["session_score"],
            marker="o"
        )
        ax_session.set_title("Session Score Over Time")
        ax_session.set_xlabel("Date")
        ax_session.set_ylabel("Weighted Score (/10)")
        plt.xticks(rotation=45)
        st.pyplot(fig_session)
    else:
        st.info("No sessions found.")

    st.markdown("---")

    # Tests grouped by session
    st.subheader("Tests Grouped by Session")

    if not sessions_df.empty:
        for _, session_row in sessions_df.iterrows():
            sid           = session_row["session_id"]
            session_score = session_row["session_score"]
            st.markdown(f"### Session {sid}")
            session_tests = dashboard_tests_df[dashboard_tests_df["session_id"] == sid]
            if not session_tests.empty:
                st.dataframe(
                    session_tests[["test_type", "score", "level", "created_at"]]
                )
                st.metric("Session Weighted Score", f"{session_score} / 10")
            st.markdown("---")
    else:
        st.info("No test sessions available.")

    # Radar chart
    st.subheader("Latest Mental Health Profile")

    latest = dashboard_tests_df.sort_values("created_at").groupby("test_type").last()
    categories = latest.index.tolist()
    values     = latest["score"].tolist()

    if len(values) >= 3:
        values += values[:1]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        fig = plt.figure()
        ax  = fig.add_subplot(111, polar=True)
        ax.plot(angles, values)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        st.pyplot(fig)
    else:
        st.info("Not enough modules for radar chart.")
    # interview responses
    st.subheader("🎥 Interview Responses")

    try:
        interview_data = api_request(
            "get",
            f"/interview/{user_id}",
            context="Loading interview history",
            timeout=DEFAULT_HISTORY_TIMEOUT,
        )
        if interview_data is None:
            st.stop()

        if interview_data:
            # Get all session numbers from interview data
            interview_session_options = [s.get('session_number') for s in interview_data if isinstance(s, dict)]
            
            if interview_session_options:
                selected_interview_session = st.selectbox(
                    "Select Session to View Interviews",
                    interview_session_options,
                    key="interview_session_selector"
                )
                
                # Filter interviews for selected session
                selected_session_interviews = next(
                    (s for s in interview_data if s.get('session_number') == selected_interview_session),
                    None
                )
                
                if selected_session_interviews:
                    interviews = selected_session_interviews.get('interviews', [])
                    st.markdown(f"### Session {selected_interview_session} - {selected_session_interviews.get('session_created_at', 'N/A')}")
                    
                    if interviews:
                        for idx, interview in enumerate(interviews, 1):
                            st.markdown(f"**Question {idx}:** {interview.get('question', 'N/A')}")
                            st.write(f"📝 **Transcript:** {interview.get('transcript', 'N/A')}")
                            st.write(f"😐 **Video Emotion:** {interview.get('video_emotion', 'N/A')}")
                            st.write(f"🎙️ **Audio Emotion:** {interview.get('audio_emotion', 'N/A')} (Confidence: {interview.get('audio_confidence', 'N/A')})")
                            st.markdown("---")
                    else:
                        st.info("No interviews in this session")
            else:
                st.info("No interview sessions found")
        else:
            st.info("No interview data found")
    except Exception as e:
        st.error(f"Error loading interview data: {str(e)}")


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("---")
st.caption("⚠️ Academic and research tool only. Not a medical diagnosis.")
