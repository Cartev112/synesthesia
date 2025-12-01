# Synesthesia BCI

A real-time brain-computer interface that generates music and visual art from EEG signals.

## Overview

Synesthesia transforms brain activity into synchronized audiovisual experiences. It uses an EEG simulator to generate realistic brain signals, processes them through a signal processing pipeline, and creates responsive music and generative art in real-time.

**Key Features:**
- Real-time EEG signal processing (simulated)
- Brain state classification (focus, relax, neutral) using ML
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

1. Open http://localhost:5173
2. Click "START SESSION" to begin
3. Watch the Hyperspace Portal visuals respond to simulated brain states
4. Adjust audio tracks and visual parameters in real-time

The simulator automatically rotates through brain states (relax → neutral → focus) every 5 seconds for demonstration.

## Architecture

```
EEG Simulator → Signal Processing → Feature Extraction → ML Classification
                                                              ↓
                                                    Music + Visual Generation
                                                              ↓
                                                    WebSocket → Frontend
```

**Pipeline Components:**
- **EEG Simulator**: Generates realistic multi-channel EEG with configurable mental states
- **Signal Processing**: Bandpass filtering, notch filtering, re-referencing
- **Feature Extraction**: Band power calculation (delta, theta, alpha, beta, gamma)
- **ML Classification**: Random Forest classifier for brain state detection, CNN for artifact rejection
- **Music Generator**: 4-layer generative system (bass, harmony, melody, texture)
- **Visual Generator**: Hyperspace Portal algorithm with parametric brain-state modulation

## Project Structure

```
synesthesia/
├── backend/
│   ├── api/              # FastAPI routes & WebSocket
│   ├── audio/            # Music synthesis (deprecated - now frontend)
│   ├── eeg/              # EEG simulator
│   ├── signal_processing/# Filtering, feature extraction
│   ├── ml/               # Random Forest & CNN classifiers, calibration
│   ├── music/            # Generative music engine
│   ├── visual/           # Visual parameter generation
│   └── pipeline/         # Real-time processing pipeline
├── frontend/
│   └── src/
│       ├── audio/        # Web Audio synthesis
│       ├── features/     # UI components
│       └── hooks/        # React hooks
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
pytest backend/tests/test_signal_processing.py
```

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

**In Progress:**
- User calibration system
- Real EEG hardware integration (OpenBCI, Muse)
- Database system

**Planned:**
- Session recording & playback
- Additional visual algorithms (Lissajous, Harmonograph, Lorenz)
- Advanced ML models (deep learning for state detection)

