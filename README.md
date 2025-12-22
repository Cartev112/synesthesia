# Synesthesia BCI

A real-time brain-computer interface that generates music and visual art from EEG signals.

## Overview

Synesthesia transforms brain activity into synchronized audiovisual experiences. It uses an EEG simulator to generate realistic brain signals, processes them through a signal processing pipeline, and creates responsive music and generative art in real-time.

**Key Features:**
- Real-time EEG signal processing (simulated or Muse S hardware)
- Brain state classification (focus, relax, neutral) using ML
- **User calibration system** for personalized state detection
- Generative ambient music engine with 4 layers
- Hyperspace Portal visual algorithm with brain-state modulation
- Low-latency WebSocket streaming
- Full-stack web interface

## Tech Stack

**Backend:**
- Python 3.10+, FastAPI, WebSockets
- Signal processing: NumPy, SciPy
- ML: scikit-learn (Random Forest), PyTorch (CNN for artifacts)
- Music: Custom generative algorithm

**Frontend:**
- React, TypeScript, Vite
- Web Audio API for music synthesis
- Canvas API for visuals
- TailwindCSS

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+

### Installation

```bash
# Clone repository
git clone <repo-url>
cd synesthesia

# Backend setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend setup
cd frontend
npm install
cd ..
```

### Running

**Backend:**
```bash
# From project root
python -m backend.api.main
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

**Frontend:**
```bash
cd frontend
npm run dev
# App: http://localhost:5173
```

## Usage

### Standard Workflow

1. Open http://localhost:5173
2. Select your device (Simulator or Muse S)
3. Complete the **calibration process** (see below)
4. Click "START SESSION" to begin
5. Watch the Hyperspace Portal visuals respond to your brain states
6. Adjust audio tracks and visual parameters in real-time

### Calibration Process

The calibration system trains a personalized model for your unique brain patterns. This significantly improves state detection accuracy.

**Calibration Stages:**

| Stage | Duration | Task |
|-------|----------|------|
| **Baseline** | 60s | Sit calmly with a neutral mindset |
| **Focus** | 60s | Count backwards from 100 by 7s (100, 93, 86...) |
| **Relax** | 60s | Close your eyes and focus on breathing |

After completing all stages, the system trains a Random Forest classifier on your data and reports validation accuracy.

**Skipping Calibration:**
- If you've already calibrated, you can skip directly to the session
- The system uses generic state detection when uncalibrated (less accurate)

## Architecture

```
EEG Device/Simulator → Signal Processing → Feature Extraction → ML Classification
                                                                       ↓
                                                         Music + Visual Generation
                                                                       ↓
                                                         WebSocket → Frontend
```

**Pipeline Components:**
- **EEG Source**: Simulator or Muse S Athena via OpenMuse
- **Signal Processing**: Bandpass filtering, notch filtering, re-referencing
- **Feature Extraction**: Band power calculation (delta, theta, alpha, beta, gamma)
- **ML Classification**: Random Forest classifier for brain state detection, CNN for artifact rejection
- **Calibration System**: Collects labeled data and trains personalized models
- **Music Generator**: 4-layer generative system (bass, harmony, melody, texture)
- **Visual Generator**: Hyperspace Portal algorithm with parametric brain-state modulation

## Calibration System

### WebSocket Protocol

The calibration system uses WebSocket messages for real-time communication:

**Client → Server:**
```javascript
// Initialize calibration (starts EEG pipeline in calibration mode)
{ type: "calibration_start", device_type: "simulator", user_id: "user123" }

// Start a calibration stage
{ type: "calibration_start_stage", stage: "baseline" }  // or "focus", "relax"

// Stop current stage
{ type: "calibration_stop_stage" }

// Train the model after all stages
{ type: "calibration_train" }

// Cancel calibration
{ type: "calibration_cancel" }
```

**Server → Client:**
```javascript
// Calibration initialized
{ type: "calibration_started", stages: ["baseline", "focus", "relax"], stage_durations: {...} }

// Stage started with instructions
{ type: "calibration_stage_started", stage: "focus", duration: 60, instructions: "Count backwards..." }

// Progress updates during stages
{ type: "calibration_progress", progress: { elapsed: 30, remaining: 30, samples_collected: 45 } }

// Stage completed
{ type: "calibration_stage_stopped", sample_counts: { neutral: 60, focus: 0, relax: 0 } }

// Training complete
{ type: "calibration_complete", validation_accuracy: 0.85, sample_counts: {...}, feature_importance: {...} }
```

### Backend API

The calibration system is implemented in `backend/ml/calibration.py`:

```python
from backend.ml.calibration import CalibrationProtocol, UserCalibration

# Create protocol for a user
protocol = CalibrationProtocol(user_id="user123")

# Start a stage
stage_info = protocol.start_stage("baseline")  # Returns instructions, duration

# Add feature samples (called automatically by pipeline)
protocol.add_sample(feature_array)

# Train after all stages
results = protocol.train_model()  # Returns accuracy, feature importance

# Apply to pipeline
pipeline.apply_calibration(protocol.calibration)
```

## Project Structure

```
synesthesia/
├── backend/
│   ├── api/              # FastAPI routes & WebSocket
│   ├── eeg/              # EEG simulator & device interfaces
│   ├── signal_processing/# Filtering, feature extraction
│   ├── ml/               # Random Forest, CNN, calibration system
│   ├── visual/           # Visual parameter generation
│   └── pipeline/         # Real-time processing pipeline
├── frontend/
│   └── src/
│       ├── audio/        # Web Audio synthesis
│       ├── features/     # UI components (visualizer, calibration, etc.)
│       └── hooks/        # React hooks (useWebSocket)
└── scripts/              # Utility scripts
```

## Configuration

Create `.env` in project root:

```env
# Application
APP_NAME=synesthesia
ENV=development

# Server
HOST=0.0.0.0
PORT=8000

# EEG
EEG_DEVICE_TYPE=simulator
EEG_SAMPLING_RATE=256
EEG_NUM_CHANNELS=8
```

## Testing

```bash
# Run all tests
pytest backend/tests/

# Specific test suites
pytest backend/tests/test_integration.py
pytest backend/tests/test_ml.py
pytest backend/tests/test_calibration.py
pytest backend/tests/test_signal_processing.py
```

### Verification Checklist

After making changes to the calibration system:

1. **Backend Pipeline Test:**
   ```bash
   pytest backend/tests/test_calibration.py -v
   ```

2. **WebSocket Integration Test:**
   - Start backend and frontend
   - Open browser console
   - Verify calibration messages are exchanged correctly

3. **Full Flow Test:**
   - Select device → Complete all 3 calibration stages → Train model
   - Verify "CALIBRATED" badge appears
   - Start session → Verify personalized model is used (check backend logs)

## Development Status

**Completed:**
- EEG simulator with realistic signals
- Signal processing pipeline
- ML brain state classification (Random Forest)
- Artifact detection (CNN)
- Real-time WebSocket streaming
- Generative music engine
- Hyperspace Portal visual algorithm
- Full-stack integration
- **User calibration system** with 3-stage protocol

**In Progress:**
- Real EEG hardware integration (OpenBCI, Muse)
- Database system for persisting calibrations

**Planned:**
- Session recording & playback
- Additional visual algorithms (Lissajous, Harmonograph, Lorenz)
- Advanced ML models (deep learning for state detection)
