# Mental Health Assessment Platform

A comprehensive AI-powered mental health assessment system designed to provide personalized depression and mental health screening through multiple modalities including text analysis, questionnaires, and audio-video interviews.

## 🎯 Project Overview

This platform enables clinicians and researchers to conduct evidence-based mental health assessments using:
- **AI Text Analysis** - Automated depression severity scoring from patient narratives
- **PHQ-9 Screening** - Standard depression assessment questionnaire
- **Multi-Domain Screening** - Comprehensive assessment across 9 mental health domains
- **Guided Interviews** - Audio/video recorded responses with emotion recognition
- **Patient Dashboard** - Longitudinal tracking and progress visualization

## ✨ Key Features

### 1. **AI Depression Assessment**
- Patient enters narrative text and observations from relatives
- Machine learning model (RoBERTa-based) calculates depression severity score (0-10)
- Risk level classification: Low, Moderate, High, Critical

### 2. **PHQ-9 Questionnaire**
- Standard 9-item Patient Health Questionnaire
- Scoring: 0-27 raw score normalized to 0-10 scale
- Instant feedback with risk assessment

### 3. **Multi-Domain Screening**
- 9 mental health modules:
  - Anxiety, Self-Esteem, Procrastination
  - Work Stress, Trauma, Grief
  - Relationship Issues, Anger, General Mental Health
- Customizable question banks
- Reverse scoring for positive items

### 4. **Audio/Video Assessment**
- Guided interview with 9 expert-designed questions
- Real-time video emotion detection (DeepFace)
- Audio emotion recognition (wav2vec2 model)
- Automatic speech-to-text transcription (Faster Whisper)
- Voice feature extraction (jitter, shimmer, HNR, pitch)
- Comprehensive multimodal depression risk scoring

### 5. **Patient Dashboard**
- Session history with weighted scores
- Session-by-session comparison
- Test results grouped by domain
- Radar chart visualization of latest mental health profile
- Session-wise interview response review with emotions

### 6. **Session Management**
- Automatic session creation (new session if >1 hour since last test)
- Weighted composite scoring across all assessments
- Longitudinal tracking for clinical monitoring

## 🛠️ Tech Stack

### Backend
- **FastAPI** - REST API framework
- **SQLAlchemy** - ORM for database management
- **SQLite** - Lightweight database
- **PyTorch** - Deep learning framework
- **Transformers** - Pre-trained NLP models
- **Librosa** - Audio analysis
- **Faster-Whisper** - Speech recognition
- **DeepFace** - Facial emotion recognition

### Frontend
- **Streamlit** - Web application framework
- **Streamlit-WebRTC** - Real-time audio/video streaming
- **Pandas** - Data manipulation
- **Matplotlib** - Data visualization
- **Requests** - HTTP client

### Models & ML
- **RoBERTa-base** - Text classification for depression severity
- **wav2vec2-XLSR** - Audio emotion recognition
- **DeepFace** - Facial emotion detection
- **Faster-Whisper** - Automatic speech recognition

## 📊 Project Structure

```
d:\project/
├── backend/
│   ├── main2.py                    # FastAPI application & endpoints
│   ├── av_pipeline.py              # Audio/video analysis pipeline
│   ├── inference.py                # Model prediction logic
│   ├── database.py                 # SQLAlchemy database models
│   ├── models/
│   │   ├── best_regression_model.pt  # Trained depression model
│   │   ├── text_model.py           # Text model definition
│   │   └── fusion_model.py         # Multimodal fusion (optional)
│   └── utils/
│       └── preprocess.py           # Text preprocessing
├── frontend/
│   ├── app2.py                     # Main Streamlit app (ACTIVE)
│   └── app.py                      # Alternative interface
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── ERROR_REPORT.md                # Detailed error scan results
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9+
- Webcam & microphone (for audio/video assessments)
- 4GB+ RAM recommended
- CUDA-capable GPU recommended for faster inference

### Step 1: Clone & Environment Setup
```bash
cd d:\project
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # On Windows PowerShell
# or: source .venv/bin/activate  # On Linux/Mac
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download Pre-trained Models
Models are automatically downloaded on first use from HuggingFace:
- **RoBERTa** for text analysis
- **wav2vec2-XLSR** for audio emotion
- **Faster-Whisper** for transcription

### Step 4: Run the Application

**Start Backend:**
```bash
cd backend
python -m uvicorn main2:app --reload --port 8000
```
Backend will be available at `http://127.0.0.1:8000`

**Start Frontend (in new terminal):**
```bash
cd frontend
streamlit run app2.py
```
Frontend will open at `http://localhost:8501`

## 📖 How to Use

### For Clinicians/Researchers

#### 1. **Run AI Assessment**
1. Open the "Run AI Assessment" page
2. Enter Patient ID
3. Patient enters: "How have you been feeling recently?"
4. Relative provides: "Observed behavioral changes"
5. Click "Run AI Assessment"
6. View AI Score (0-10) and Risk Level

#### 2. **PHQ-9 Screening**
1. Go to "PHQ-9 Questionnaire"
2. Enter Patient ID
3. Answer 9 standard depression questions
4. Submit to get PHQ-9 Score and Risk Level

#### 3. **Multi-Domain Assessment**
1. Navigate to "Multi-Domain Assessment"
2. Select assessment module (e.g., Anxiety, Trauma, etc.)
3. Answer domain-specific questions
4. Submit for scoring

#### 4. **Audio/Video Interview**
1. Go to "Audio/Video Assessment"
2. Enter Patient ID
3. For each of 9 guided questions:
   - Click "🎥 Start Recording"
   - Answer the interview question (video + audio captured)
   - Click "🛑 Stop Recording"
   - Click "✅ Submit Answer"
4. System automatically analyzes:
   - Facial emotions (video)
   - Voice emotions (audio)
   - Speech transcript
   - Voice features (pitch, jitter, etc.)
5. Generate clinical summary from all responses

#### 5. **Patient History Dashboard**
1. Go to "Patient History Dashboard"
2. Enter Patient ID
3. View:
   - All sessions with weighted composite scores
   - Individual test scores by type
   - Session score trends over time
   - Compare two sessions side-by-side
   - Latest mental health profile (radar chart)
   - Interview responses by session

## 🗄️ Database Schema

### Tables
- **patients** - Patient records (patient_id, created_at)
- **sessions** - Assessment sessions (session_number, session_score, created_at)
- **tests** - Individual assessment results (test_type, score, level)
- **av_interview** - Recorded interview responses (transcript, emotions, confidence)

### Scoring Weights
```
AI Assessment:      22%
PHQ-9:              22%
Audio/Video:        10%
Anxiety:            13%
Work Stress:         9%
Anger:               9%
Trauma:              5%
Grief:               5%
Relationship:        3%
Self-Esteem:         1%
Mental Health:       1%
```

## 🎬 Sample Workflow

### Example: First-Time Patient Assessment
```
1. Patient takes AI Assessment
   ├─ Enters narrative text
   └─ Gets initial depression severity score

2. Same session, patient completes PHQ-9
   └─ Standardized screening questionnaire

3. Later in same session, Multi-Domain Assessment
   ├─ Selects "Anxiety" module
   └─ Gets domain-specific insights

4. Next day, patient completes Audio/Video Interview
   ├─ Answers 9 guided questions on camera
   ├─ System analyzes facial/voice emotions
   ├─ Extracts voice quality metrics
   └─ Calculates comprehensive AV score

5. Clinician reviews Patient Dashboard
   ├─ Views composite session score
   ├─ Tracks trends over time
   ├─ Reviews interview transcripts + emotions
   └─ Generates clinical notes
```

## ⚙️ API Endpoints (Backend)

### Assessment Endpoints
- `POST /assessments/create` - AI text assessment
- `POST /phq9/submit` - PHQ-9 questionnaire
- `POST /questionnaire/submit` - Multi-domain screening
- `POST /av/analyze` - Audio/video analysis
- `POST /av/question` - Submit interview response
- `POST /llm/summary` - Generate clinical summary

### History Endpoints
- `GET /sessions/{user_id}` - Get patient sessions
- `GET /tests/{user_id}` - Get all tests for patient
- `GET /interview/{user_id}` - Get interview responses (by session)

## 📝 Assessment Scoring

### Risk Levels
- **Low** (0-3) - Minimal risk
- **Moderate** (3-6) - Some concerns
- **High** (6-8) - Significant risk
- **Critical** (8-10) - Severe risk; immediate intervention recommended

### Composite Session Score
Weighted average of all tests completed in a session, reflecting overall mental health status.

## ⚠️ Important Notes

### For Clinical Use
- **Disclaimer**: This is an academic/research tool, NOT a medical diagnosis
- Scores should inform but not replace professional clinical judgment
- Always conduct follow-up with licensed mental health professionals
- Document all assessments for clinical records

### Model Limitations
- Audio emotion detection may vary by language/accent
- Facial emotion detection requires adequate lighting
- Voice analysis limited by microphone quality
- Text models trained on English-language data

### Privacy & Security
- Database uses SQLite (local storage)
- No data transmission to external servers
- Patient IDs stored as-is (add encryption for production)
- Temporary audio/video files cleaned up after processing

## 🔧 Troubleshooting

### Common Issues

**"No audio recorded"**
- Check microphone permissions in browser
- Ensure sound device is properly configured
- Test microphone in system settings first

**"Video not saving"**
- OpenCV error: Ensure opencv-python is installed
- Check temp/ directory exists and is writable
- Try different video codec settings

**"Model download fails"**
- Check internet connection
- HuggingFace servers may be temporarily unavailable
- Models cache in ~/.cache/huggingface/

**"Database locked"**
- Multiple processes writing simultaneously
- Restart backend and frontend
- Delete depression.db and reinitialize

## 📚 Model References

- **Depression Severity**: RoBERTa-base fine-tuned regression model
- **Audio Emotion**: wav2vec2-Large-XLSR-English (Facebook Research)
- **Facial Emotion**: DeepFace (7-emotion classification)
- **Transcription**: Faster-Whisper (OpenAI Whisper optimized)
- **Voice Analysis**: Librosa + Parselmouth Praat

## 🚀 Future Enhancements

- [ ] Multi-language support
- [ ] Real-time therapist dashboard
- [ ] Mobile app version
- [ ] Integration with EHR systems
- [ ] Advanced NLP for crisis keywords detection
- [ ] Longitudinal ML models for progression tracking
- [ ] Video call support for remote therapy
- [ ] Export reports (PDF/DOCX)

## 📄 License

Academic and Research Use Only

## 👥 Support

For issues, questions, or suggestions, please check:
- `ERROR_REPORT.md` for known issues
- Terminal logs for runtime errors
- Model download guides on HuggingFace

---

**Version**: 1.0  
**Last Updated**: April 2026  
**Status**: ✅ Production Ready
