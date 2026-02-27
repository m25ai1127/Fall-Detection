# Fall Detection Project — Weekly Progress Report

## Week 1: Research & Project Setup
**Dates: Feb 10 - Feb 16**

### Research & Literature Review
- Studied existing fall detection approaches (wearable sensors, depth cameras, vision-based)
- Identified gap: most systems need expensive depth sensors (Kinect, RealSense)
- Proposed solution: Use AI-based depth estimation (MiDaS) with a single regular camera
- Selected datasets: UR Fall Detection (URFD) — 70 video sequences (30 falls + 40 normal)
- Selected tools: MiDaS (depth), MediaPipe (pose), PyTorch LSTM (classifier)

### Project Structure Created
- Initialized project with modular architecture
- Created `config.py` — centralized configuration (all hyperparameters in one file)
- Set up `requirements.txt` with all dependencies
- Created `.gitignore` for large files exclusion
- Directory structure: `src/models/`, `src/data/`, `src/utils/`, `docs/`, `results/`

### Core Model Development
- Implemented `depth_estimator.py` — MiDaS wrapper with depth normalization and noise filtering
- Implemented `pose_estimator.py` — Multi-backend pose detection (MediaPipe + OpenCV fallback)
- Implemented `feature_extractor.py` — 177-dimensional feature vector extraction per frame
  - 132: Pose landmarks (33 joints × 4 values: x, y, z, visibility)
  - 33: Depth values at each joint location
  - 3: Bounding box ratio, distance-to-floor, center of mass height
  - 9: Vertical velocity of 9 key joints

### Deliverables
- Complete project structure with 13+ Python modules
- README.md with architecture and methodology

---

## Week 2: Data Pipeline & Training
**Dates: Feb 16 - Feb 19**

### Dataset Preparation
- Built `download_dataset.py` — automated downloader for URFD dataset with progress bars
- Downloaded initial 10 sequences (5 falls + 5 normal) for testing pipeline
- Built `preprocess.py` — processes video frames through MiDaS → MediaPipe → Feature Extractor

### Model Architecture
- Implemented `fall_detector.py` — Bidirectional LSTM with Attention mechanism
  - Input: 177 features × 30 frames (1 second window)
  - Architecture: 2-layer BiLSTM (hidden=128) + Attention + Fully Connected
  - Total parameters: 745,060
  - Output: fall probability (0.0 to 1.0)

### Training Pipeline
- Built `train.py` with early stopping, LR scheduling, checkpointing
- Built `evaluate.py` with full metrics (accuracy, sensitivity, specificity, F1, AUC-ROC)
- Built `dataset.py` — PyTorch DataLoader with sliding window sampling (stride=10)
- Trained initial model on 10 sequences → 100% accuracy (small dataset)

### Inference System
- Built `src/inference.py` — supports video files and live webcam
- Real-time visualization: skeleton overlay, depth map, confidence bar, fall alerts

### Deliverables
- Working training pipeline (train → evaluate → inference)
- Initial model with 100% accuracy on small test set (21 samples)
- Training curves, confusion matrix, ROC curve plots

---

## Week 3: Documentation & Presentation
**Dates: Feb 19 - Feb 25**

### Added Fall Confirmation Window
- New feature: System waits 3 seconds before confirming a fall
- 3 visual states: Normal (green) → Suspected (orange) → Confirmed (red)
- If person recovers within 3 seconds → alert cancelled (reduces false alarms)
- Helps distinguish exercise/stumbles from real falls

### Documentation
- Created `docs/study_guide.md` — comprehensive explanation of all concepts
- Created `docs/study_notes.md` — quick-reference notes for all config parameters
- Created `docs/progress_report.md` — formal progress report
- Created `docs/six_week_plan.md` — complete 6-week project timeline

### Presentation
- Generated PowerPoint presentation using python-pptx
- 12 slides covering: problem statement, architecture, dataset, modules, timeline

### Deliverables
- Fall confirmation window feature (config: FALL_CONFIRMATION_SECONDS=3)
- Complete documentation package
- PowerPoint presentation for professor meeting

---

## Week 4: Full Dataset Training & Evaluation
**Dates: Feb 25 - Feb 26**

### Full Dataset
- Downloaded complete URFD dataset: 140 files (30 falls + 40 ADL, each with RGB + depth)
- Preprocessed all 70 video sequences (~13,000+ frames total)
- Feature extraction: MiDaS depth + pose estimation on every frame
- Created 1,041 training samples (231 falls, 810 normal) using sliding window

### Retrained Model
- Retrained LSTM on full dataset (728 train, 156 validation, 157 test samples)
- Training completed in 20 seconds, stopped at epoch 20 (early stopping)
- Learning rate reduced from 0.001 → 0.0005 at epoch 17 (LR scheduler)

### Final Evaluation Results

| Metric | Initial (10 sequences) | Full Dataset (70 sequences) |
|--------|:----------------------:|:---------------------------:|
| Accuracy | 100% | **99.4%** |
| Sensitivity (Recall) | 100% | **97.1%** |
| Specificity | 100% | **100%** |
| Precision | 100% | **100%** |
| F1 Score | 1.0 | **0.9855** |
| False Alarm Rate | 0% | **0%** |
| AUC-ROC | 1.0 | **1.0** |
| Test Samples | 21 | 157 |

### Confusion Matrix
```
                Predicted
                Normal    Fall
Actual Normal [  122  |    0  ]    (zero false alarms)
Actual Fall   [    1  |   34  ]    (1 missed fall out of 35)
```

### Key Observations
- 99.4% accuracy on 157 test samples — much more reliable than initial 100% on 21 samples
- Only 1 missed fall out of 35 → 97.1% sensitivity
- Zero false alarms → 100% specificity
- Results validate the approach: MiDaS depth + pose + LSTM works effectively

### Bug Fixes
- Fixed Unicode encoding errors on Windows (cp1252 charset limitations)
- Fixed MediaPipe compatibility issue with Python 3.13 (added multi-backend fallback)

### Deliverables
- Retrained model on full URFD dataset
- Updated evaluation plots (confusion matrix, ROC curve, training curves)
- Evaluation metrics JSON

---

## Summary: Project Status After 4 Weeks

### Completed
- Full project codebase (13+ Python modules)
- Complete data pipeline (download → preprocess → train → evaluate → inference)
- 99.4% accuracy on full URFD dataset (157 test samples)
- Live camera support with fall confirmation window
- Documentation and presentation materials

### Planned for Weeks 5-6
- Hyperparameter tuning and cross-validation
- NTU RGB+D dataset integration (exercise activities)
- Transformer-based model comparison
- Add acceleration features to improve trajectory analysis
- Final project report and updated presentation
