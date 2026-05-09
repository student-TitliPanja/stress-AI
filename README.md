
---

## 🎯 Overview

This system performs **real-time stress detection** using two synchronized modalities:

1. **Facial Analysis**: 17 Facial Action Unit (AU) intensities extracted via OpenFace 3.0
2. **Speech Analysis**: 84-dimensional acoustic features (Pitch, Jitter, Shimmer, MFCCs)

Both streams are fused using a **Cross-Modal Attention Transformer** and output a continuous **Valence-Arousal (V-A) regression** — the clinical gold standard for affective computing.

Stress is detected as a **personalized deviation** from each user's calibrated neutral baseline.

---

## 🚀 Key Innovations

| # | Innovation | Description |
|---|-----------|-------------|
| 1 | **Objective AU Detection** | Biomechanical muscle movement analysis via OpenFace 3.0 (not subjective emotion labels) |
| 2 | **Full Speech Pipeline** | Complete audio modality with Pitch, Jitter, Shimmer, 40 MFCCs, Delta MFCCs, RMS |
| 3 | **Continuous V-A Regression** | 2D Valence-Arousal output instead of discrete emotion classes |
| 4 | **Cross-Modal Attention** | Multimodal Transformer where face attends to speech & vice versa |
| 5 | **Personal Calibration + FL** | 90-second neutral baseline per user; all data processed locally |

---

## 🛠 Tech Stack

| Component | Tool/Library |
|-----------|-------------|
| Language | Python 3.10+ |
| Deep Learning | TensorFlow 2.x / Keras |
| Computer Vision | OpenCV (Haar Cascade face detection) |
| Facial AU Analysis | OpenFace 3.0 (with simulation fallback) |
| Audio Analysis | librosa |
| Web Framework | Flask (Jinja2 templates) |
| Frontend | HTML5, CSS3, Bootstrap 5, JavaScript |
| Data/Viz | Pandas, NumPy, Matplotlib, Seaborn |
| Federated Learning | Flower (flwr) |

---

## 📁 Project Structure

```
app/
├── app.py                    # Main Flask application (all routes)
├── model_training.py         # Full model training script
├── openface_pipeline.py      # OpenFace AU extraction wrapper
├── audio_pipeline.py         # Real-time audio feature extraction
├── calibration.py            # Personal baseline management
├── analysis_dashboard.py     # 7 dashboard visualizations
├── federated_client.py       # Flower FL client
├── full_model.h5             # Trained model (generated)
├── emotion_log.csv           # Session V-A + stress log
├── users.json                # User database
├── requirements.txt          # Dependencies
├── baselines/                # Per-user calibration data (local)
├── templates/
│   ├── base.html             # Base template with navbar/styling
│   ├── index.html            # Home page
│   ├── register.html         # Registration with validation
│   ├── login.html            # Login page
│   ├── admin.html            # Admin user management
│   ├── calibration.html      # 90s baseline calibration
│   ├── live.html             # Live dual-stream analysis
│   ├── upload.html           # Static file analysis
│   └── analysis.html         # Full analytics dashboard
└── static/
    ├── css/
    ├── js/
    ├── assets/
    └── uploads/              # Uploaded files for analysis
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10+ installed
- pip package manager
- Webcam (optional, for live analysis)
- Microphone (optional, for speech analysis)

### Installation Steps

```bash
# 1. Navigate to the app directory
cd app

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Train the model
python model_training.py

# 5. Run the application
python app.py
```

### OpenFace 3.0 (Optional)

For real AU extraction (instead of simulation):

1. Download OpenFace from: https://github.com/TadasBaltrusaitis/OpenFace
2. Build following their instructions
3. Set `openface_path` in the pipeline to point to the `FeatureExtraction` binary

---

## 🚀 Running the Application

```bash
cd app
python app.py
```

The server starts at: **http://localhost:5000**

### Default Admin Credentials

- **Email**: `admin@stress.ai`
- **Password**: `Admin123`

---

## 📖 Usage Guide

### 1. Register & Login
- Create an account at `/register`
- Password requirements: 8+ chars, 1 uppercase, 1 lowercase, 1 digit

### 2. Calibrate Your Baseline
- Navigate to `/calibrate`
- Sit relaxed for 90 seconds while the system records your neutral state
- Read the neutral passage aloud for speech baseline

### 3. Live Analysis
- Go to `/live` and click "Start Analysis"
- The system captures webcam + audio in real-time
- View your stress score, V-A position, and zone-based recommendations

### 4. Upload Analysis
- Go to `/upload` to analyze static images or audio files
- Supported formats: JPG, PNG (images), WAV, MP3, OGG (audio)

### 5. Analytics Dashboard
- Navigate to `/analysis` for comprehensive visualizations
- View V-A scatter plots, stress timelines, zone distributions, and more

---

## 📊 Dashboard Views

The analytics dashboard includes **7 visualizations**:

1. **V-A Scatter Plot** — Points colored by zone, neutral baseline marked
2. **Stress Score Timeline** — Line chart with 0.5 threshold
3. **Rolling Average Stress** — 20-second sliding window
4. **Zone Distribution Pie Chart** — Time proportion per zone
5. **Daily Average Stress** — Bar chart by date
6. **Valence & Arousal Timelines** — Stacked line charts
7. **Acoustic Feature Trends** — Pitch, Jitter, Shimmer sub-plots

---

## 🔒 Federated Learning

**Privacy Guarantee**: All biometric data stays on your local device.

- No raw webcam frames leave the device
- No raw audio leaves the device
- Only anonymized model gradients can be shared (optional)

```bash
# Start FL server
python federated_client.py --server

# Start FL client (on each device)
python federated_client.py --client
```

---

## 📚 Datasets

| Dataset | Modalities | Usage |
|---------|-----------|-------|
| Ulm-TSST | Audio, Video, Physio | Primary training (real induced stress) |
| StressID | Audio, Video, Physio | Primary training (large-scale) |
| MAHNOB-HCI | Audio, Video, EEG, ECG | V-A fine-tuning (gold standard) |
| AVEC | Audio, Video | Benchmark evaluation |
| RAVDESS | Audio, Video | Pre-training only |
| FER-2013 | Images only | AU backbone pre-training only |

> ⚠️ Do NOT use FER-2013 alone — it lacks audio and uses discrete mislabeled emotion classes.

---

## 📈 Performance Targets

| Metric | Target |
|--------|--------|
| AU-LSTM facial model CCC | ≥ 0.70 |
| Speech LSTM model CCC | ≥ 0.65 |
| Multimodal fusion CCC (V-A) | ≥ 0.80 |
| Face detection latency | < 100ms per frame |
| Live cam frame rate | ≥ 15 fps |
| Calibration duration | 90 seconds |

---

## 📖 References

1. Naveen Sundar Kumar P et al. (2024). Realtime Face Emotion Recognition. IJARCCE Vol.13
2. D. Sirisha et al. (2024). Real-Time Stress Detection Using Facial Emotion Recognition. IJARIIE Vol.10
3. Devanshu Shah et al. (2021). Real-Time Facial Emotion Recognition. IEEE GCAT
4. Titli Panja & Soumya Ranjan Pradhan (2025). Real-Time Stress Detection. IEM PRJCS381
5. Giannakakis G. et al. (2017). Stress detection using facial cues. BSPC 31
6. Zhang J. et al. (2020). Video-based stress detection. Sensors 20(19)
7. Giannakakis G. et al. (2020). Automatic stress detection via AUs. IEEE FG
8. Giannakakis G. et al. (2019). Psychological stress detection. IEEE TAC 13(1)
9. Almeida J. & Rodrigues F. (2021). FER for Stress Detection. ICEIS
10. Sinha S. & Sharma A. (2023). Stress Alarm Raiser. IJIRCST 11(3)

---

## 📝 License

This project is developed for academic purposes as part of Innovative Project-I (PRJCS381) at IEM. All rights reserved.
