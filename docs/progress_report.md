# Project Progress Report
## AI-Based Fall and Abnormal Motion Detection Using Depth-Assisted Single-Camera

**Date:** February 2026  
**Status:** Full Dataset Training Complete (Week 4)

---

## 1. Work Completed

### 1.1 Literature Review and Methodology Design
- Reviewed existing fall detection approaches using RGB-D cameras, wearable sensors, and vision-based methods.
- Designed a methodology that uses monocular depth estimation (MiDaS) to simulate depth data from a regular single camera, combined with pose estimation for skeletal joint extraction.
- Chose LSTM-based temporal classification for distinguishing fall events from normal activities.

### 1.2 Dataset Selection and Download
- Selected the **UR Fall Detection Dataset (URFD)** — 70 sequences total (30 falls + 40 ADL).
- Downloaded and processed all 70 sequences (30 falls + 40 ADL).
- Identified **NTU RGB+D** as a supplementary dataset for future work.

### 1.3 System Architecture
Developed a complete software pipeline:

| Module | Description | Status |
|--------|-------------|--------|
| Depth Estimator | MiDaS-based monocular depth map generation | Completed & Tested |
| Pose Estimator | Skeletal joint extraction (33 landmarks) | Completed & Tested |
| Feature Extractor | Combined depth + pose features (177-dim vector) | Completed & Tested |
| LSTM Classifier | Bidirectional LSTM with attention mechanism | Completed & Tested |
| Data Pipeline | Download, preprocessing, PyTorch data loader | Completed & Tested |
| Training Script | Training loop with early stopping, LR scheduling | Completed & Tested |
| Evaluation Script | All metrics from methodology (Section 4.5) | Completed & Tested |
| Inference Script | Video processing with visualization overlay | Completed |

### 1.4 Feature Engineering (Section 4.3)
- Pose landmarks — 33 joints x 4 values (x, y, z, visibility) = 132 features
- Depth values at joint locations = 33 features
- Bounding box aspect ratio — posture change indicator
- Distance-to-floor estimation — derived from depth
- Center of mass height — weighted average of joint positions
- Vertical velocity of key joints — sudden downward movement detection

### 1.5 Training Results
- **Dataset:** 70 URFD sequences (30 falls + 40 ADL), 13,000+ frames total
- **Samples:** 1,041 sliding-window sequences (231 falls, 810 normal)
- **Split:** 728 train / 156 validation / 157 test
- **Model:** Bidirectional LSTM, 745,060 parameters
- **Training:** 20 epochs (early stopping triggered)
- **Training time:** ~20.3 seconds on CPU
- **Best validation loss:** 0.0444

### 1.6 Evaluation Results (Section 4.5)

| Metric | Value |
|--------|-------|
| Accuracy | 99.4% |
| Sensitivity (Recall) | 97.1% |
| Specificity | 100.0% |
| Precision | 100.0% |
| F1-Score | 0.9855 |
| False Alarm Rate | 0.0% |
| AUC-ROC | 1.0000 |

**Confusion Matrix (157 test samples):**
- True Negatives (Normal correctly classified): 122
- True Positives (Falls correctly detected): 34
- False Positives: 0
- False Negatives: 1

Plots generated: Training curves, Confusion matrix, ROC curve — available in `results/` folder.

---

## 2. Issues Faced and Solutions

### 2.1 No Physical Depth Camera
**Problem:** Project requires depth data but no depth camera (Kinect, RealSense) was available.  
**Solution:** Used publicly available URFD dataset which includes Kinect depth streams. Integrated MiDaS monocular depth estimation to generate pseudo-depth from RGB — this is the core of the "depth-assisted single-camera" approach.

### 2.2 MediaPipe Compatibility with Python 3.13
**Problem:** Latest MediaPipe versions dropped the `mp.solutions.pose` API, and the new Tasks API has C-binding issues on Python 3.13/Windows.  
**Solution:** Implemented a multi-backend pose estimator that auto-selects the best available backend:
1. MediaPipe solutions API (Python <= 3.12)
2. MediaPipe Tasks API (newer versions)
3. OpenCV HOG-based person detector with heuristic landmarks as fallback

### 2.3 Dataset Server Speed
**Problem:** URFD hosted on a Polish university server with limited bandwidth (~80-150 KB/s).  
**Solution:** Download script with progress bars, resume capability, and option to limit sequence count for testing.

### 2.4 PyTorch API Deprecation
**Problem:** `ReduceLROnPlateau(verbose=True)` removed in PyTorch 2.x.  
**Solution:** Removed deprecated parameter from scheduler initialization.

### 2.5 Windows Encoding
**Problem:** Unicode characters (checkmarks) caused `UnicodeEncodeError` on Windows cp1252 terminal.  
**Solution:** Replaced with ASCII-safe equivalents in output.

---

## 3. Pending Work (Weeks 5-6)

### Week 5: Improvements
- [ ] Hyperparameter tuning (learning rate, hidden size, sequence length)
- [ ] Cross-validation to verify model robustness
- [ ] Download NTU RGB+D subset for exercise activities
- [ ] Add acceleration features for trajectory analysis

### Week 6: Final Report and Submission
- [ ] Write final project report
- [ ] Create demonstration video showing end-to-end pipeline
- [ ] Compare results with existing methods from literature
- [ ] Final presentation slides (updated with full results)

---

## 4. Project Structure

```
fall-detection/
├── README.md                    # Project overview and usage guide
├── requirements.txt             # Python dependencies
├── config.py                    # All hyperparameters and settings
├── src/
│   ├── data/
│   │   ├── download_dataset.py  # Dataset download utility
│   │   ├── preprocess.py        # Feature extraction pipeline
│   │   └── dataset.py           # PyTorch Dataset class
│   ├── models/
│   │   ├── depth_estimator.py   # MiDaS depth estimation
│   │   ├── pose_estimator.py    # Pose landmark detection
│   │   ├── feature_extractor.py # Combined feature engineering
│   │   └── fall_detector.py     # LSTM classifier
│   ├── utils/
│   │   ├── visualization.py     # Plotting utilities
│   │   └── helpers.py           # Common utilities
│   ├── train.py                 # Model training
│   ├── evaluate.py              # Model evaluation
│   └── inference.py             # Video inference
├── results/                     # Output plots and metrics
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── training_curves.png
│   └── evaluation_metrics.json
└── models/                      # Trained model weights
    └── best_model.pth
```

## 5. GitHub Repository
**Link:** https://github.com/m25ai1127/fall-detection

## 6. References
1. B. Kwolek and M. Kepski, "Human fall detection on embedded platform using depth maps and wireless accelerometer," *Comp. in Biology and Medicine*, vol. 53, pp. 245-254, 2014.
2. A. Shahroudy et al., "NTU RGB+D: A large scale dataset for 3D human activity analysis," in *CVPR*, 2016.
3. R. Ranftl et al., "Towards Robust Monocular Depth Estimation," *IEEE TPAMI*, 2022.
4. C. Lugaresi et al., "MediaPipe: A Framework for Building Perception Pipelines," *arXiv:1906.08172*, 2019.
