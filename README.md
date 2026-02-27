# AI-Based Fall and Abnormal Motion Detection Using Depth-Assisted Single-Camera

<p align="center">
  <img src="docs/architecture.png" alt="Architecture Diagram" width="700">
</p>

## Overview

This project implements an **AI-based fall detection system** that uses **monocular depth estimation** from a single RGB camera combined with **human pose estimation** to detect falls and abnormal motion patterns. The system processes video frames through two parallel pipelines:

1. **MiDaS Depth Estimation** — generates pseudo-depth maps from RGB frames (simulating a depth camera)
2. **MediaPipe Pose Estimation** — extracts 33 body skeletal landmarks per frame

The depth values are mapped to skeletal joints to create **3D posture information**, from which temporal features are extracted and fed into an **LSTM classifier** for fall/normal activity classification.

## Architecture

```
Video Frame ──┬──> MiDaS Depth Estimation ──> Depth Map
              │                                   │
              └──> MediaPipe Pose Estimation ──> Joints ──> Feature Fusion
                                                                │
                                          Temporal Sequence Buffer
                                                                │
                                          LSTM Fall Classifier ──> Fall / Normal
```

### Feature Extraction (per frame)
- **Vertical motion characteristics** — sudden downward movement of key joints
- **Bounding box aspect ratio** — width/height ratio indicating posture change
- **Distance-to-floor estimation** — depth values relative to ground plane
- **Joint depth values** — sampled from MiDaS output at each landmark
- **Joint velocities** — frame-to-frame displacement of key joints

## Datasets

### Primary: UR Fall Detection Dataset (URFD)
- **70 sequences** (30 falls + 40 activities of daily living)
- RGB + Depth streams from Microsoft Kinect
- Source: [University of Rzeszów](http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html)
- Reference: Kwolek & Kepski, *Computers in Biology and Medicine*, 2014

### Supplementary: NTU RGB+D
- Large-scale RGB-D activity dataset
- Action A43 (falling down) + normal activities used for supplementary training
- Source: [ROSE Lab, NTU Singapore](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
- Reference: Shahroudy et al., *CVPR*, 2016

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended, not required)

### Setup
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/fall-detection-depth-assisted.git
cd fall-detection-depth-assisted

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Download Dataset
```bash
# Download URFD dataset
python -m src.data.download_dataset --dataset urfd

# Or manually place dataset files in data/raw/urfd/
```

### 2. Preprocess Data
```bash
# Extract features from all sequences
python -m src.data.preprocess --dataset urfd --output data/processed/

# Quick test with a few samples
python -m src.data.preprocess --dataset urfd --max-sequences 5
```

### 3. Train Model
```bash
# Full training
python -m src.train --epochs 50 --batch-size 16

# Quick smoke test
python -m src.train --epochs 2 --subset 10
```

### 4. Evaluate Model
```bash
# Run evaluation on test set
python -m src.evaluate --model models/best_model.pth
```

### 5. Run Inference on Video
```bash
# Process a video file
python -m src.inference --input path/to/video.avi --output results/output.mp4

# Process with visualization
python -m src.inference --input path/to/video.avi --output results/output.mp4 --show-depth --show-skeleton
```

## Project Structure

```
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config.py                    # Central configuration
├── .gitignore                   # Git ignore rules
├── data/
│   ├── raw/                     # Downloaded datasets (git-ignored)
│   └── processed/               # Preprocessed features (git-ignored)
├── models/                      # Saved model weights (git-ignored)
├── results/                     # Output videos, plots, metrics
├── docs/                        # Documentation and diagrams
├── src/
│   ├── data/
│   │   ├── download_dataset.py  # Dataset download utility
│   │   ├── preprocess.py        # Feature extraction pipeline
│   │   └── dataset.py           # PyTorch Dataset class
│   ├── models/
│   │   ├── depth_estimator.py   # MiDaS depth estimation wrapper
│   │   ├── pose_estimator.py    # MediaPipe pose estimation wrapper
│   │   ├── feature_extractor.py # Combined feature extraction
│   │   └── fall_detector.py     # LSTM fall classifier
│   ├── utils/
│   │   ├── visualization.py     # Plotting and visualization
│   │   └── helpers.py           # Common utilities
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   └── inference.py             # Video inference script
└── notebooks/                   # Jupyter notebooks (optional)
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall classification performance |
| **Sensitivity (Recall)** | Ability to correctly detect fall events |
| **Specificity** | Ability to correctly identify normal activities |
| **False Alarm Rate** | Incorrect fall detections during normal activities |
| **F1-Score** | Harmonic mean of precision and recall |
| **AUC-ROC** | Area under the ROC curve |

## Methodology

This project follows the methodology described in Section 4 of the project report:

1. **Data Sources (§4.1)** — URFD + NTU RGB+D datasets
2. **Data Pre-processing (§4.2)** — Temporal sync, depth normalization, noise filtering
3. **Pose Estimation & Feature Extraction (§4.3)** — MediaPipe + MiDaS depth-assisted features
4. **Temporal Analysis & Classification (§4.4)** — LSTM-based sequence classification
5. **Evaluation (§4.5)** — Accuracy, Sensitivity, False Alarm Rate

## References

1. M. Mubashir et al., "A survey on fall detection: Principles and approaches," *Neurocomputing*, 2013.
2. C. Rougier et al., "Robust video surveillance for fall detection based on 3D shape analysis," *IEEE TCSVT*, 2011.
3. S. Gasparrini et al., "Proposal and experimental characterization of an ad-hoc fall log for depth camera-based fall detection systems," *IEEE Sensors Journal*, 2015.
4. S. Khan and I. Ahmad, "DeepFall: Non-invasive fall detection with spatio-temporal autoencoders," *arXiv:1808.01234*, 2018.
5. A. Jalal et al., "Depth video-based human activity recognition system," *IEEE TCE*, 2012.
6. J. Han et al., "Enhanced computer vision with Microsoft Kinect sensor: A review," *IEEE Trans. Cybernetics*, 2013.
7. B. Kwolek and M. Kepski, "Human fall detection on embedded platform using depth maps and wireless accelerometer," *CBM*, 2014.
8. A. Shahroudy et al., "NTU RGB+D: A large scale dataset for 3D human activity analysis," *CVPR*, 2016.

## License

This project is developed for academic purposes. Datasets are used under their respective licenses (CC BY-NC-SA 4.0 for URFD).

## Author

Vignesh — Final Year Project, 2025-2026
