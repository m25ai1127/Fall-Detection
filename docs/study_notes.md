# Fall Detection Project — Complete Study Notes
> Everything you need to know, topic by topic

---

## 1. Python Basics Used

### `import os`
- `os` = Operating System module. Lets Python talk to your computer.
- `os.path.join("data", "raw")` → creates path `data\raw`
- `os.makedirs("results")` → creates a folder
- `os.path.exists("file.txt")` → checks if file exists

### `from pathlib import Path`
- Modern way to handle file paths (cleaner than `os.path`).
- `Path("data") / "raw" / "video.mp4"` → creates `data\raw\video.mp4`
- `Path(__file__).parent` → folder where the current file lives
- `PROJECT_ROOT = Path(__file__).parent` → all paths are relative to project root, so it works no matter where you move the folder.

---

## 2. MiDaS Depth Estimation

### Three model sizes:
| Model | Speed | Accuracy | Use case |
|-------|-------|----------|----------|
| **MiDaS_small** (ours) | Fast | Good | Real-time on CPU |
| DPT_Hybrid | Medium | Better | ResNet + Transformer combined |
| DPT_Large | Slow | Best | Full Transformer, needs GPU |

### DPT = Dense Prediction Transformer
- **Dense** = predicts a value for EVERY pixel
- **Prediction** = outputs depth values
- **Transformer** = same architecture behind ChatGPT, but for images
- **Hybrid** = CNN (ResNet) at bottom + Transformer at top

### Processing time on CPU:
| Model | Per frame | Live camera |
|-------|-----------|-------------|
| MiDaS_small | ~0.05s | ~5-8 FPS |
| DPT_Hybrid | ~0.3s | ~1-2 FPS |
| DPT_Large | ~0.8s | <1 FPS (unusable) |

### Device auto-detection:
```python
MIDAS_DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
```
Checks if NVIDIA GPU exists → uses GPU (fast), otherwise CPU (slower but works).

---

## 3. Pre-processing Settings

### Frame size: `640 × 480`
All video frames resized to same size for consistency and speed.

### Depth normalization: `(0.0, 1.0)`
MiDaS outputs random large numbers → normalize to 0-1 range.
- 0.0 = closest to camera
- 1.0 = farthest from camera

### Median Filter (kernel size = 5)
- **NOT a real kernel** — just a sliding window (5×5 pixels)
- Collects all 25 values in the window, sorts them, picks the MIDDLE value
- Removes noise (random wrong pixels) from the depth map
- Better than average filter because it completely ignores outliers
- **Median vs Average:** Average of [50,50,50,250,50] = 90 (bad). Median = 50 (good).

---

## 4. Pose Estimation (MediaPipe)

### Model complexity: `0=lite, 1=full, 2=heavy`
We use 1 (full) — good balance of speed and accuracy.

### Confidence thresholds: `0.5`
- Detection confidence: "Only detect a person if ≥50% sure"
- Tracking confidence: "Keep tracking if ≥50% sure it's the same person"

### 33 Landmarks
Each body part has ONE point with a unique ID (0-32):
- 0=nose, 7=left ear, 8=right ear
- 11=left shoulder, 12=right shoulder
- 23=left hip, 24=right hip
- 25=left knee, 26=right knee
- 27=left ankle, 28=right ankle
- (plus eyes, mouth, wrists, fingers, feet = 33 total)

### 4 values per landmark: (x, y, z, visibility)
- **x** = horizontal position (0=left, 1=right)
- **y** = vertical position (0=top, 1=bottom)
- **z** = relative depth WITHIN the body (arm in front vs behind — relative to hip center)
- **visibility** = how confident the detection is (0=hidden, 1=clearly visible)

### 9 KEY_JOINTS (most important for fall detection):
nose, left/right shoulder, left/right hip, left/right knee, left/right ankle

---

## 5. Pose z vs MiDaS Depth — Why Both?

| | Pose z | MiDaS Depth |
|--|--------|-------------|
| **Reference** | Body-relative (hip = 0) | Camera-relative |
| **Tells you** | Which limb is in front/behind | How far person is from camera |
| **Example** | "Hand is in front of hip" | "Person is 2.5 meters away" |
| **Falls toward camera** | z doesn't change much | Depth decreases → detected! |

Both give DIFFERENT information. Combining them = better fall detection.

---

## 6. Feature Vector (177 dimensions)

### Formula:
```
FEATURE_DIM = 33×4 + 33 + 3 + 9 = 177
```

| Part | Count | What it is |
|------|:-----:|------------|
| Pose landmarks (x,y,z,vis × 33) | 132 | Body posture |
| Depth at each joint | 33 | Distance from camera |
| Bounding box aspect ratio | 1 | Body shape (tall vs wide) |
| Distance to floor | 1 | How high hips are from ground |
| Center of mass height | 1 | Average height of ALL joints |
| Vertical velocity (9 key joints) | 9 | Speed of downward movement |
| **Total** | **177** | |

### Bounding Box Aspect Ratio
- `width ÷ height` of box around the person
- Standing: ratio ≈ 0.4 (tall, narrow)
- Fallen: ratio ≈ 3.0+ (wide, short — lying flat)

### Distance to Floor
- `1.0 - average_hip_y`
- Standing: ~0.5 (hips at mid-frame)
- Fallen: ~0.15 (hips near ground)

### Center of Mass Height
- Average y value of ALL 33 joints
- Differs from floor distance: floor distance only checks hips, center of mass checks ENTIRE body
- Helps distinguish bending down (head still high) from falling (everything low)

### Vertical Velocity
- `current_frame.y - previous_frame.y` for each key joint
- **Most important feature!**
- Fall: large positive values (fast downward movement)
- Normal: near-zero values
- Key difference: falling happens FAST, sitting happens SLOWLY — velocity captures this

---

## 7. Temporal Sequence

| Setting | Value | Meaning |
|---------|-------|---------|
| FPS | 30 | Camera captures 30 frames per second |
| SEQUENCE_LENGTH | 30 | LSTM looks at 30 frames (= 1 second) per decision |
| SEQUENCE_STRIDE | 10 | Sliding window moves by 10 frames (creates overlapping samples) |

### Stride = 10:
- Window slides by 10 frames each time
- Creates overlapping training samples
- A fall appears in multiple samples → model learns it from different perspectives

---

## 8. LSTM Model Settings

| Setting | Value | Meaning |
|---------|-------|---------|
| Hidden size | 128 | Memory capacity — 128 numbers of "understanding" |
| Layers | 2 | Two stacked LSTMs — basic patterns → complex patterns |
| Dropout | 0.3 | Randomly turns off 30% neurons to prevent memorization |
| Bidirectional | True | Reads sequence forward AND backward |

### Bidirectional:
- Forward: "I know what happened before each frame"
- Backward: "I also know what happens after"
- Combined: much better at understanding the full motion

---

## 9. Training Settings

| Setting | Value | Meaning |
|---------|-------|---------|
| Batch size | 16 | Process 16 samples at once per training step |
| Learning rate | 0.001 | Size of adjustment steps (not too big, not too small) |
| Weight decay | 0.0001 | Prevents model from overcomplicating — keeps rules simple |
| Max epochs | 50 | Maximum training rounds (stopped at 25 due to early stopping) |
| Early stopping patience | 10 | Stop if no improvement for 10 epochs in a row |
| LR scheduler patience | 5 | Cut learning rate in half if no improvement for 5 epochs |
| LR scheduler factor | 0.5 | Multiply learning rate by 0.5 when reducing |
| Train/Val/Test split | 70/15/15 | 70% for learning, 15% for checking progress, 15% for final test |
| Random seed | 42 | Makes results reproducible (tradition from Hitchhiker's Guide) |

### Classification:
- **NUM_CLASSES = 1**: Single output (0.0 to 1.0) — binary problem
- **FALL_THRESHOLD = 0.5**: If output ≥ 0.5 → FALL, else → Normal
- Can adjust: 0.3 for safety-critical (catch all falls), 0.7 for low false alarms

---

## 10. Inference Settings

### Fall Confirmation Window (NEW feature):
```
Person goes down → "FALL SUSPECTED" (orange, 3s wait)
    → Person stays down 3 seconds → "FALL CONFIRMED!" (red alert)
    → Person gets back up → "Normal" (cancelled, no alert)
```

| Setting | Value | Meaning |
|---------|-------|---------|
| FALL_CONFIRMATION_SECONDS | 3 | Wait 3 seconds before confirming fall |
| RECOVERY_THRESHOLD | 0.3 | If confidence drops below 30%, person recovered |
| ALERT_COOLDOWN_SECONDS | 5 | After confirmed fall, wait 5 seconds before next alert |

### Gap between thresholds (0.3 to 0.5):
Prevents flickering — confidence jumping between 0.48 and 0.52 won't keep triggering/cancelling alerts.

---

## 11. Results Explained

### Test set: 21 samples (7 falls, 14 normal)

| Metric | Value | What it means |
|--------|:-----:|---------------|
| Accuracy | 100% | All 21 predictions correct |
| Precision | 100% | When it says "FALL", it's always right |
| Sensitivity/Recall | 100% | It catches ALL falls |
| Specificity | 100% | It correctly identifies ALL normal activities |
| F1 Score | 100% | Perfect balance of precision and recall |
| False Alarm Rate | 0% | Never cries wolf |
| AUC-ROC | 1.0 | Overall quality = perfect |

### Confusion matrix values:
- True Positives (TP) = 7 → was fall, said fall ✅
- True Negatives (TN) = 14 → was normal, said normal ✅
- False Positives (FP) = 0 → no false alarms
- False Negatives (FN) = 0 → no missed falls

### ⚠️ Why 100% is misleading:
- Small test set (only 21 samples) — model may have memorized patterns
- Expected realistic accuracy on full dataset: **90-95%**
- Need more training data and cross-validation for trustworthy results

---

## 12. Known Limitations & Professor Answers

### Exercise detection:
- Current model may confuse push-ups/sit-ups with falls (never trained on exercise videos)
- Solution: add NTU RGB+D exercise data in weeks 4-5

### Live camera performance:
- Runs at ~2-5 FPS on CPU (MiDaS + pose estimation per frame)
- GPU would make it smoother (~15-30 FPS)

### Why MiDaS instead of real depth camera?
- Eliminates need for expensive hardware (Kinect, RealSense)
- Works with ANY regular webcam
- Key innovation of this project

### Is it a replica?
- NO — unique combination of MiDaS + 33 pose landmarks + 177-dim features + Bidirectional LSTM with attention
- Uses standard tools (MiDaS, MediaPipe, PyTorch) combined in an original way
- This is exactly how real research works — build on existing tools, combine them innovatively

---

## 13. Project Files Summary

| File | Purpose |
|------|---------|
| `config.py` | All settings in one place |
| `src/models/depth_estimator.py` | MiDaS depth map generation |
| `src/models/pose_estimator.py` | 33-joint skeleton detection |
| `src/models/feature_extractor.py` | Creates 177-dim feature vector |
| `src/models/fall_detector.py` | LSTM classifier with attention |
| `src/data/download_dataset.py` | Downloads URFD dataset |
| `src/data/preprocess.py` | Processes videos → features |
| `src/data/dataset.py` | PyTorch data loader with sliding window |
| `src/train.py` | Training loop with early stopping |
| `src/evaluate.py` | Computes metrics, generates plots |
| `src/inference.py` | Live camera + video file detection |
| `src/utils/visualization.py` | Plotting helpers |
| `src/utils/helpers.py` | Common utilities |

### Pipeline flow:
```
Frame → MiDaS (depth) → MediaPipe (pose) → Feature Extractor (177 numbers)
    → LSTM (30 frames) → Confirmation Window (3s) → FALL or NORMAL
```
