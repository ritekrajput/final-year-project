import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import time
import threading
import cv2

# ─────────────────────────────────────────────
# AUDIO + VIDEO RECORDER
# Records mic audio via sounddevice in a background thread (no ffmpeg needed)
# ─────────────────────────────────────────────

AUDIO_SR = 16000   # 16 kHz mono — what Whisper & wav2vec2 expect

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
                if elapsed > 30:
                    self.recording = False
                else:
                    self.frames.append(img.copy())
        return frame


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Mental Health Assessment", layout="centered")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

page = st.sidebar.radio(
    "Navigation",
    [
        "Run AI Assessment",
        "PHQ-9 Questionnaire",
        "Multi-Domain Assessment",
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
            response = requests.post(
                f"{BASE_URL}/assessments/create",
                json={
                    "user_id": user_id,
                    "patient_text": patient_text,
                    "relative_text": relative_text,
                }
            )
            result = response.json()
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
        "Little interest or pleasure in doing things?",
        "Feeling down, depressed, or hopeless?",
        "Trouble falling/staying asleep, or sleeping too much?",
        "Feeling tired or having little energy?",
        "Poor appetite or overeating?",
        "Feel bad about yourself or that you are a failure or have let yourself or your family down?",
        "Have trouble concentrating on things, such as reading, work, or watching television?",
        "Have you been moving or speaking so slowly that other people have noticed, or the opposite — being fidgety or restless?",
        "Thoughts you'd be better off dead or hurting yourself?",
    ]

    options = ["Not at all (0)", "Several days (1)",
               "More than half (2)", "Nearly every day (3)"]

    answers = {}
    for i, q in enumerate(questions):
        choice = st.radio(q, options, key=f"phq_{i}")
        answers[f"q{i+1}"] = options.index(choice)

    if st.button("Submit PHQ-9"):
        if not user_id:
            st.error("Patient ID required.")
        else:
            response = requests.post(
                f"{BASE_URL}/phq9/submit",
                json={"user_id": user_id, "answers": answers}
            )
            result = response.json()
            st.success("Submitted")
            st.metric("PHQ-9 Score", f"{result['score']} / 10")
            st.metric("Risk Level", result["level"])


# ─────────────────────────────────────────────
# PAGE 3 — MULTI-DOMAIN
# ─────────────────────────────────────────────

elif page == "Multi-Domain Assessment":

    st.title("Multi-Domain Screening")

    user_id = st.text_input("Patient ID")

    module = st.selectbox(
        "Select Module",
        ["anxiety", "self_esteem", "procrastination",
         "workstress", "trauma", "grief",
         "relationship", "anger", "mental_health"]
    )

    question_bank = {
        "anxiety": [
            "Feeling nervous, anxious, or on edge?",
            "Not being able to stop or control worrying?",
            "Worrying too much about different things?",
            "Trouble relaxing?",
            "Being so restless that it's hard to sit still?",
            "Becoming easily annoyed or irritable?",
            "Feeling afraid as if something awful might happen?",
        ],
        "self_esteem": [
            "I feel confident in my abilities.",
            "I feel that I am a person of worth.",
            "I often feel useless or inadequate.",
            "I compare myself negatively to others.",
            "I feel proud of achievements.",
            "I doubt myself even when I succeed.",
        ],
        "procrastination": [
            "I delay starting important tasks.",
            "I wait until the last minute to complete assignments.",
            "I feel overwhelmed and avoid responsibilities.",
            "Distract yourself?",
            "Regret delaying?",
        ],
        "workstress": [
            "Overwhelmed?",
            "Deadline pressure?",
            "Emotionally exhausted?",
            "Performance pressure?",
            "Job insecurity?",
        ],
        "trauma": [
            "Have you experienced a distressing or traumatic event?",
            "Do you have intrusive memories about it?",
            "Do you avoid reminders of the event?",
            "Do you feel emotionally numb?",
            "Do you feel constantly on guard or easily startled?",
        ],
        "grief": [
            "Intense sadness?",
            "Difficulty accepting loss?",
            "Life feels meaningless?",
            "Avoid reminders?",
            "Feel guilt?",
        ],
        "relationship": [
            "I feel emotionally supported in my relationship.",
            "Feel misunderstood?",
            "Communication with my partner/family is difficult.",
            "Relationship anxiety?",
            "Conflicts remain unresolved for long periods.",
        ],
        "anger": [
            "Lose temper easily?",
            "Feel intense anger over small issues.",
            "Regret anger?",
            "Hard to calm down?",
            "Does your anger affect your relationships, work, or daily life?",
        ],
        "mental_health": [
            "Feel emotionally stable?",
            "Motivated daily?",
            "Hopeful about future?",
            "Struggle to enjoy activities?",
            "Overwhelmed emotionally?",
            "Socially connected?",
        ],
    }

    reverse_items = {
        "self_esteem":  [0, 1],
        "mental_health":[0, 1, 2, 5],
        "relationship": [0],
    }

    options = ["Not at all (0)", "Several days (1)",
               "More than half the time (2)", "Nearly every day (3)"]

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

            response = requests.post(
                f"{BASE_URL}/questionnaire/submit",
                json={"user_id": user_id, "module": module, "answers": answers}
            )
            result = response.json()
            st.success("Submitted")
            st.metric(f"{module.upper()} Score", f"{result['score']} / 10")
            st.metric("Risk Level", result["level"])


# ─────────────────────────────────────────────
# PAGE 4 — AUDIO / VIDEO ASSESSMENT
# ─────────────────────────────────────────────

elif page == "Audio/Video Assessment":

    st.title("Guided Interview Assessment")

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
        st.session_state.recording_stopped = False

    if user_id:

        q_idx = st.session_state.current_q

        if q_idx < len(questions):

            st.subheader(f"Question {q_idx+1}")
            st.write(questions[q_idx])

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

            col1, col2 = st.columns(2)

            if ctx.video_processor:

                if col1.button("🎥 Start Recording"):
                    ctx.video_processor.start_recording()
                    st.session_state.recording_stopped = False
                    st.session_state.saved_video = None
                    st.session_state.saved_audio = None
                    st.success("📹 Recording started...")

                if col2.button("🛑 Stop Recording"):
                    ctx.video_processor.stop_recording()

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
                            st.success(f"✅ Recording stopped and saved! ({len(frames)} frames captured)")
                        else:
                            st.error("Failed to save recording")
                    else:
                        st.warning("⚠️ No frames captured")

            if st.button("✅ Submit Answer"):

                # Check if recording was stopped and saved
                if not st.session_state.recording_stopped:
                    st.error("❌ Please record and stop before submitting!")
                elif st.session_state.saved_video and st.session_state.saved_audio:
                    try:
                        # Send files to backend for processing and database storage
                        with open(st.session_state.saved_video, "rb") as video_f, \
                             open(st.session_state.saved_audio, "rb") as audio_f:
                            files = {
                                "video": ("video.avi", video_f, "video/x-msvideo"),
                                "audio": ("audio.wav", audio_f, "audio/wav")
                            }
                            data = {
                                "user_id": user_id,
                                "question": questions[q_idx]
                            }
                            
                            # Send to backend endpoint that handles saving to database
                            response = requests.post(
                                f"{BASE_URL}/av/question",
                                files=files,
                                data=data
                            )
                            result = response.json()

                        if "error" not in result:
                            st.session_state.responses.append(result)
                            st.session_state.current_q += 1
                            st.session_state.recording_stopped = False
                            st.success("✅ Answer submitted and saved to database!")
                            st.rerun()
                        else:
                            st.error(f"Backend error: {result['error']}")
                    except Exception as e:
                        st.error(f"Error submitting answer: {str(e)}")
                else:
                    st.error("❌ No recording found! Please record first.")

        else:
            st.success("✅ Interview Completed!")

            if st.button("Generate Clinical Summary"):
                response = requests.post(
                    f"{BASE_URL}/llm/summary",
                    json={"responses": st.session_state.responses}
                )
                st.write(response.json()["summary"])

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
    try:
        tests_response    = requests.get(f"{BASE_URL}/tests/{user_id}")
        sessions_response = requests.get(f"{BASE_URL}/sessions/{user_id}")
        tests_data    = tests_response.json()
        sessions_data = sessions_response.json()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

    if not tests_data:
        st.warning("No test records found.")
        st.stop()

    tests_df = pd.DataFrame(tests_data)
    tests_df["created_at"] = pd.to_datetime(tests_df["created_at"])
    tests_df = tests_df.sort_values("created_at")

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
        session_tests = tests_df[tests_df["session_id"] == session_id]

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
            session_tests = tests_df[tests_df["session_id"] == sid]
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

    latest = tests_df.sort_values("created_at").groupby("test_type").last()
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
        interview_response = requests.get(f"{BASE_URL}/interview/{user_id}")
        interview_data = interview_response.json()

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