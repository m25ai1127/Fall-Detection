# Understanding Your Fall Detection Project — Study Guide

Read this before your professor meeting. It explains EVERYTHING in simple language.

---

## THE BIG PICTURE

**Problem:** Old people fall at home and nobody notices. We want a camera system that can automatically detect when someone falls and raise an alert.

**Your Solution:** Instead of using an expensive depth camera (like Kinect), you use a REGULAR camera and use AI to estimate the depth. Then you combine depth + body pose to detect falls.

**Why this is clever:** Real depth cameras are expensive. Your system works with any regular webcam or CCTV camera.

---

## HOW IT WORKS (Step by Step)

Think of it like a pipeline — data flows through each step:

```
Camera Frame (RGB image)
        │
        ├──→ Step 1: MiDaS (Depth Estimation)
        │         └── Creates a depth map: "how far is each pixel from camera"
        │
        ├──→ Step 2: Pose Estimation
        │         └── Detects 33 body joints (head, shoulders, hips, knees, ankles...)
        │
        └──→ Step 3: Feature Extraction
                  │   Combines depth + pose into a number vector (177 numbers)
                  │
                  └──→ Step 4: LSTM (Brain of the system)
                            └── Looks at 30 frames in a row, decides: FALL or NORMAL
```

---

## EXPLAINING EACH MODULE

### 1. MiDaS Depth Estimator (`depth_estimator.py`)

**What it does:** Takes a regular photo and estimates how far each pixel is from the camera.

**How to explain it:**
> "MiDaS is a pre-trained neural network by Intel. It takes an RGB image and outputs a depth map — a grayscale image where darker pixels are closer to the camera and lighter pixels are farther away. I used the lightweight MiDaS_small variant because it's fast enough for real-time use."

**Why we need it:**
> "Fall detection works better with depth information because we can track how a person's height decreases when they fall. With just 2D images, it's hard to tell if someone is falling or just sitting down. Depth gives us the 3rd dimension."

**Key term:** Monocular depth estimation = estimating depth from a SINGLE camera image (mono = one, ocular = eye).

---

### 2. Pose Estimator (`pose_estimator.py`)

**What it does:** Finds the human body in the image and extracts 33 joint positions (like placing dots on shoulders, elbows, hips, knees, ankles).

**How to explain it:**
> "I use pose estimation to detect 33 skeletal landmarks on the human body. Each landmark has an x, y position (where it is in the image), a z value (relative depth), and a visibility score (how confident the detection is)."

**The 33 joints include:** nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles, heels, toes.

**Why we need it:**
> "Falls have a specific pattern — the hip joints rapidly move downward, the body orientation changes from vertical to horizontal, and the center of mass drops. By tracking joint positions, we can capture these patterns."

---

### 3. Feature Extractor (`feature_extractor.py`)

**What it does:** Combines depth and pose into ONE number vector (177 numbers) that describes the person's posture in that frame.

**The 177 features breakdown:**

| Feature | Count | What it captures |
|---------|-------|-----------------|
| Pose landmarks (x, y, z, visibility × 33 joints) | 132 | Body posture |
| Depth values at each joint | 33 | How far each body part is from camera |
| Bounding box aspect ratio | 1 | Width/height ratio — standing vs lying |
| Distance to floor | 1 | How high the person is off the ground |
| Center of mass height | 1 | Average height of all joints |
| Vertical velocities (9 key joints) | 9 | How fast joints are moving downward |
| **Total** | **177** | |

**How to explain it:**
> "I designed a 177-dimensional feature vector that captures everything relevant for fall detection. The key insight is combining spatial features (where the body is) with depth features (how far it is) and temporal features (how fast things are changing). The vertical velocity features are particularly important — during a fall, joints accelerate downward rapidly."

**Key features for fall detection:**
- **Bounding box aspect ratio:** A standing person has a tall, narrow box (ratio < 1). A fallen person has a wide, short box (ratio > 1). This single number tells you a lot about posture.
- **Distance to floor:** If this drops suddenly, the person is falling.
- **Vertical velocity:** During a normal activity velocity is low. During a fall, it spikes.

---

### 4. LSTM Classifier (`fall_detector.py`)

**What it does:** Looks at 30 frames worth of features (30 × 177 numbers) and decides: FALL or NORMAL.

**How to explain it:**
> "I use a Bidirectional LSTM network for temporal classification. LSTM stands for Long Short-Term Memory — it's a type of recurrent neural network designed to learn patterns in sequences. It's bidirectional because it looks at frames both forward and backward in time, which helps capture the complete fall motion."

**Architecture details (if professor asks):**
- Input: 30 frames × 177 features
- Bidirectional LSTM: processes sequence in both directions
- 2 layers, hidden size 128
- Attention mechanism: learns which frames in the sequence are most important
- Output: single probability (0 = normal, 1 = fall)
- Total parameters: 745,060

**Why LSTM and not a simple neural network?**
> "A fall is NOT a single-frame event. You can't look at one image and say 'this is a fall'. You need to see the SEQUENCE — the person was standing, then they started falling, then they're on the ground. LSTM is designed exactly for this — learning patterns in sequences over time."

**What is the Attention mechanism?**
> "Not all frames in the sequence are equally important. The frames during the actual fall contain the most useful information. The attention mechanism automatically learns to focus on the most important frames and ignore the less relevant ones."

---

## THE DATASET

### UR Fall Detection Dataset (URFD)

**What it is:** A publicly available dataset recorded at the University of Rzeszow in Poland.

**How to explain it:**
> "URFD contains 70 video sequences — 30 falls and 40 normal daily activities (ADL). Each sequence is recorded using a Microsoft Kinect camera which provides both RGB and depth streams. I use 10 sequences for initial development and will expand to the full dataset."

**Why this dataset:**
> "URFD is widely used in fall detection research. It provides both RGB and depth data, which lets me validate my monocular depth estimation against real Kinect depth. The paper by Kwolek and Kepski (2014) is a standard reference in this field."

---

## TRAINING PROCESS

**How to explain it:**
> "I split the data into training (70%), validation (15%), and test (15%). The model trains for multiple epochs — each epoch means one pass through the entire training data. I use early stopping, which means training automatically stops when the model stops improving on the validation set. This prevents overfitting."

**Key training concepts (if professor asks):**
- **Epoch:** One complete pass through all training data
- **Early stopping:** Stop training when validation loss stops decreasing (patience = 10 epochs)
- **Learning rate scheduling:** If loss plateaus, the learning rate is reduced by half to fine-tune
- **Gradient clipping:** Prevents the gradients from exploding (common in LSTMs)

**Your results:**
- Trained for 25 epochs (early stopping kicked in)
- Training accuracy: 100%
- Validation accuracy: 100%
- These are on a small dataset — will need more data for realistic results

---

## EVALUATION METRICS (Section 4.5 of your methodology)

**Know these definitions:**

| Metric | Meaning | Your Score |
|--------|---------|------------|
| **Accuracy** | Overall correct predictions / total predictions | 100% |
| **Sensitivity (Recall)** | Falls correctly detected / total actual falls | 100% |
| **Specificity** | Normal activities correctly identified / total normal | 100% |
| **Precision** | Correct fall predictions / total fall predictions | 100% |
| **F1-Score** | Balance between precision and recall (harmonic mean) | 1.0 |
| **False Alarm Rate** | Normal activities incorrectly flagged as falls | 0% |
| **AUC-ROC** | Area Under ROC Curve — overall model quality | 1.0 |

**Confusion Matrix (what it means):**
```
                 Predicted Normal    Predicted Fall
Actual Normal         14 (TN)           0 (FP)
Actual Fall            0 (FN)           7 (TP)
```
- **TN (True Negative):** Correctly said "not a fall" — 14 times
- **TP (True Positive):** Correctly detected a fall — 7 times
- **FP (False Positive):** Wrongly said "fall" when it wasn't — 0 times
- **FN (False Negative):** Missed a real fall — 0 times

---

## COMMON QUESTIONS YOUR PROFESSOR MIGHT ASK

**Q: Why did you use MiDaS instead of a real depth camera?**
> "The goal was depth-ASSISTED single-camera, which means the system should work with any regular camera. MiDaS provides monocular depth estimation, removing the need for expensive hardware like Kinect or RealSense cameras."

**Q: Why LSTM instead of CNN or Transformer?**
> "Falls are temporal events — they happen over a sequence of frames. LSTM is specifically designed for sequential data. I chose LSTM over Transformers because the sequence length is short (30 frames) and LSTM is more efficient for short sequences."

**Q: Your accuracy is 100% — isn't that suspicious?**
> "Yes, the current dataset is small (10 sequences, 136 samples). The fall and normal patterns are quite distinct in this subset. I expect accuracy to settle around 90-95% when I train on the full 70-sequence URFD dataset, which will have more variation and edge cases."

**Q: How does the system handle real-time video?**
> "The inference script processes video frame-by-frame. MiDaS_small runs fast enough for real-time on a GPU. On CPU it's slower but still functional. The system maintains a buffer of the last 30 frames and classifies them continuously."

**Q: What is the feature dimension 177?**
> "It's 33 joints × 4 values (132) + 33 depth values + 1 bounding box ratio + 1 floor distance + 1 center of mass height + 9 vertical velocities = 177."

**Q: What pre-trained models did you use?**
> "MiDaS_small for depth estimation (pre-trained by Intel on mixed datasets) and a pose landmarker model for body joint detection. Both are used as feature extractors — the LSTM classifier is trained from scratch on the fall detection data."

---

## FILE-BY-FILE QUICK REFERENCE

If your professor opens any file, here's what to say:

- **config.py** — "This is the central configuration. All hyperparameters, paths, and settings are defined here so they're easy to change without modifying code."
- **depth_estimator.py** — "This loads MiDaS and runs depth estimation on each frame."
- **pose_estimator.py** — "This detects the human skeleton — 33 body joints."
- **feature_extractor.py** — "This combines pose + depth into the 177-dim feature vector."
- **fall_detector.py** — "This is the LSTM model that classifies fall vs normal."
- **download_dataset.py** — "This downloads the URFD dataset from the university server."
- **preprocess.py** — "This runs the full pipeline: load frames → depth → pose → features → save."
- **dataset.py** — "This creates training batches using sliding windows over the sequences."
- **train.py** — "Training loop with early stopping and learning rate scheduling."
- **evaluate.py** — "Computes all the metrics and generates plots."
- **inference.py** — "Processes a video file and produces annotated output with fall alerts."

---

## KEY TERMS TO REMEMBER

- **Monocular** = single camera
- **RGB-D** = color image + depth
- **LSTM** = Long Short-Term Memory (sequential neural network)
- **Bidirectional** = processes sequence forward AND backward
- **Attention** = mechanism to focus on important frames
- **Epoch** = one pass through all training data
- **Early stopping** = stop when model stops improving
- **Sliding window** = take overlapping chunks of 30 frames
- **ADL** = Activities of Daily Living (normal activities)
- **Ground truth** = the correct labels (human-marked)
- **AUC-ROC** = Area Under ROC Curve (model quality)
