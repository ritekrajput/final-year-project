# Multi-Modal Machine Learning System for Early Detection of Mental Health Conditions

This repository contains the implementation of a research-oriented mental health screening platform inspired by the project paper and presentation for a multimodal clinical decision-support system. The system combines standardized psychiatric screening with AI-based analysis of text, audio, and video signals to estimate psychological distress on a continuous severity scale.

## Project Summary

Conventional mental health screening often depends on self-reported questionnaires and semi-structured interviews, both of which are vulnerable to cognitive bias and intentional masking of symptoms. This project addresses that limitation by combining:

- free-text depression assessment using a RoBERTa-based regression model
- standardized psychiatric screening such as PHQ-9 and anxiety-related modules
- audio-derived behavioral cues such as pitch, jitter, shimmer, loudness, and speech rate
- video-derived cues such as facial emotion distributions over time
- longitudinal patient-session tracking through a normalized relational database

The intended research architecture is a multimodal learning pipeline in which text, audio, and video features are fused through a cross-attention mechanism to produce a continuous mental health severity score.

## Research Objectives

The system is designed around the following research goals:

- build an early screening platform for mental health distress using multiple modalities instead of relying on only one source of information
- combine clinically validated questionnaire signals with AI-derived behavioral features
- estimate continuous severity rather than only a binary label
- support longitudinal tracking across repeated sessions
- preserve a human-in-the-loop workflow suitable for clinical decision support

## Research Architecture

The paper describes the system as a three-layer architecture.

### 1. Presentation Layer

Implemented in Streamlit, the frontend provides:

- AI free-text assessment
- PHQ-9 questionnaire
- multi-domain screening
- audio/video interview workflow
- historical dashboard and visual summaries

### 2. Business Logic Layer

Implemented in FastAPI, the backend is responsible for:

- receiving assessment data from the frontend
- routing requests to text, audio, and video inference pipelines
- computing composite clinical scores
- managing patient sessions
- serving historical records and dashboard data

### 3. Persistence Layer

Implemented using SQLite and SQLAlchemy ORM, the database stores:

- patient identities
- assessment sessions
- test-level scores and risk levels
- interview transcripts and emotion outputs

## Core Modalities

### Text Modality

The text pipeline is based on `RoBERTa-base` and follows the paper’s design:

- input text is tokenized with a maximum sequence length of `256`
- token embeddings are passed through `RoBERTa-base`
- the `[CLS]` representation is extracted as the global semantic feature
- dropout `0.4` is applied
- a linear regression head predicts a continuous severity score on a `1-10` scale

This pathway is intended to capture implicit indicators of psychological distress that may not be fully expressed through structured questionnaires.

### Audio Modality

The audio pipeline extracts behavioral and paralinguistic cues from speech, including:

- pitch
- jitter
- shimmer
- harmonic-to-noise ratio
- loudness / energy
- MFCC-derived descriptors
- pause duration
- speech rate

The platform also performs:

- speech transcription using Faster-Whisper
- audio emotion classification using wav2vec2-based speech emotion modeling

### Video Modality

The video pipeline processes interview recordings frame by frame to derive:

- dominant facial emotion
- temporal emotion distributions
- negative-emotion prevalence
- structured video emotion feature vectors

The current implementation uses DeepFace-based facial emotion analysis to approximate the visual branch described in the paper.

### Multimodal Fusion

The research paper describes a cross-attention fusion strategy in which:

- text embeddings are aligned against audio embeddings
- text embeddings are aligned against video embeddings
- fused multimodal features are passed into a regression head / MLP
- the output is a continuous mental health severity score

The repository includes a multimodal fusion model scaffold reflecting this architecture in:

- [backend/models/fusion_model.py](/C:/Users/Sayan%20Chakraborty/Desktop/BEP/final-year-project/backend/models/fusion_model.py)

## Clinical Scoring Engine

In addition to model-based inference, the system computes a session-level weighted composite score using completed clinical modules. The scoring logic is dynamically normalized so skipped modules do not unfairly skew the final score.

### Assessment Weights

The research paper emphasizes the following weighting strategy:

| Module | Weight |
|---|---:|
| AI Assessment (Text) | 25% |
| PHQ-9 | 25% |
| Anxiety / GAD-7 | 15% |
| Work Stress | 10% |
| Anger | 10% |
| Trauma | 5% |
| Grief | 5% |
| Relationship | 3% |
| Self-Esteem | 1% |
| General Mental Health | 1% |

### Risk Thresholds

The final score is interpreted using four clinical risk bands:

| Score Range | Risk Level |
|---|---|
| 0-3 | Low |
| 3-6 | Moderate |
| 6-8 | High |
| 8-10 | Critical |

## Database Design

The system follows a normalized relational structure to support longitudinal psychiatric monitoring.

### Tables

- `patients`
  - unique patient record
- `sessions`
  - one-hour grouped assessment sessions
- `tests`
  - individual module scores and risk levels
- `av_interview`
  - interview transcript and audio/video emotion outputs

This design supports repeated assessment over time and session-wise comparison in the dashboard.

## Repository Structure

```text
final-year-project/
├── backend/
│   ├── main2.py
│   ├── av_pipeline.py
│   ├── database.py
│   ├── inference.py
│   ├── models/
│   │   ├── feature_schema.py
│   │   ├── fusion_model.py
│   │   ├── text_model.py
│   │   └── best_regression_model.pt            # expected after training
│   ├── services/
│   │   └── multimodal_service.py
│   ├── training/
│   │   ├── train_text_regressor.py
│   │   └── train_multimodal_regressor.py
│   └── utils/
│       └── preprocess.py
├── frontend/
│   ├── app.py
│   └── app2.py
├── model/
│   └── model.py
├── requirements.txt
├── run_backend.ps1
├── ERROR_REPORT.md
└── README.md
```

## Installation

### Prerequisites

- Python 3.9+
- webcam and microphone for AV assessment
- internet access for first-time model downloads when required
- optional CUDA-capable GPU for faster inference/training

### Environment Setup

```powershell
cd C:\Users\Sayan Chakraborty\Desktop\BEP\final-year-project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the System

### Backend

Recommended from project root:

```powershell
.\run_backend.ps1
```

Alternative:

```powershell
python .\backend\main2.py
```

### Frontend

```powershell
cd .\frontend
python -m streamlit run app2.py
```

## Training the Research Models

Two training scripts are included for generating the checkpoint files expected by the backend.

### Text Regression Checkpoint

Generates:

- `backend/models/best_regression_model.pt`

Expected CSV fields:

- `patient_text`
- `relative_text`
- `score`

Run:

```powershell
python -m backend.training.train_text_regressor --data .\data\text_dataset.csv
```

### Multimodal Fusion Checkpoint

Generates:

- `backend/models/best_multimodal_model.pt`

Expected CSV fields include:

- `patient_text`
- `relative_text`
- `score`
- engineered audio feature columns
- engineered video feature columns

Run:

```powershell
python -m backend.training.train_multimodal_regressor --data .\data\multimodal_dataset.csv
```

## Backend Endpoints

### Core Assessment Endpoints

- `POST /assessments/create`
- `POST /phq9/submit`
- `POST /questionnaire/submit`
- `POST /av/analyze`
- `POST /av/question`
- `POST /multimodal/assess`
- `POST /llm/summary`

### History Endpoints

- `GET /sessions/{user_id}`
- `GET /tests/{user_id}`
- `GET /interview/{user_id}`

## Important Research Notes

This repository should be interpreted as a research and academic system, not as a production medical platform.

- It is a clinical decision-support tool, not a diagnostic replacement.
- Human review remains necessary for all high-risk outcomes.
- Any critical-risk output should be reviewed by a qualified mental health professional.
- The reported paper metrics depend on trained datasets and checkpoints, not only on architecture definition.

## Current Implementation Notes

The repository now includes code structures for the paper’s multimodal design, including:

- RoBERTa-based text regression
- audio and video feature extraction
- cross-attention fusion model scaffold
- training scripts for text and multimodal checkpoints

However, exact reproduction of the paper’s reported performance requires:

- the original labeled training data
- the final trained checkpoint files
- the same train/validation protocol used during experiments

## Ethical Position

This project follows the paper’s intended human-in-the-loop and ethical-AI framing:

- support early screening, not automated diagnosis
- preserve clinician oversight
- reduce dependence on single-modality self-reporting
- keep outputs interpretable through continuous scores and risk bands

## References

Primary references cited by the paper include work on:

- multimodal depression detection with cross-attention
- RoBERTa-based text regression for mental health severity
- audio-visual depression detection from facial and vocal features
- ethical AI in computational psychiatry

For the full write-up and citation list, refer to:

- [RP.pdf](C:/Users/Sayan%20Chakraborty/Desktop/RP.pdf)
- [77_35Presentation.pdf](C:/Users/Sayan%20Chakraborty/Desktop/77_35Presentation.pdf)
- [77_35Abstract.pdf](C:/Users/Sayan%20Chakraborty/Desktop/77_35Abstract.pdf)
