# Six-Week Project Plan
## AI-Based Fall and Abnormal Motion Detection Using Depth-Assisted Single-Camera

---

### Week 1 (Feb 17 - Feb 23): Environment Setup and Data Acquisition

**Objectives:**
- Set up the development environment (Python, PyTorch, OpenCV, MediaPipe)
- Study the UR Fall Detection Dataset (URFD) structure and download sequences
- Read and understand the dataset paper by Kwolek & Kepski (2014)

**Tasks:**
- Install Python virtual environment and required packages
- Write the dataset download script with progress tracking
- Download URFD dataset (fall and ADL sequences with RGB + depth pairs)
- Explore the data — understand frame format, resolution, depth encoding
- Set up Git repository and initial project structure

**Deliverables:**
- Working development environment
- Downloaded dataset (minimum 10 sequences)
- Initial project structure committed to GitHub

---

### Week 2 (Feb 24 - Mar 2): Depth Estimation and Pose Estimation Modules

**Objectives:**
- Implement monocular depth estimation using MiDaS
- Implement human pose landmark extraction
- Begin the data preprocessing pipeline

**Tasks:**
- Research MiDaS architecture and select appropriate model variant (MiDaS_small for speed)
- Write the depth estimation module — load model, run inference, normalize output
- Implement noise filtering and depth map normalization (Section 4.2)
- Write the pose estimation module — extract 33 body landmarks per frame
- Test both modules on sample frames from the URFD dataset
- Handle edge cases: missing depth, no person detected, partial visibility

**Deliverables:**
- Working depth estimator generating depth maps from RGB input
- Working pose estimator extracting 33 skeletal joints
- Sample visualization showing depth maps and skeleton overlays

---

### Week 3 (Mar 3 - Mar 9): Feature Engineering and Data Pipeline

**Objectives:**
- Design and extract the feature vector combining depth and pose (Section 4.3)
- Build the complete preprocessing pipeline
- Create the PyTorch dataset and data loader

**Tasks:**
- Implement feature extraction module with all features from methodology:
  - Pose landmarks (33 joints x 4 values = 132 features)
  - Depth values at joint locations (33 features)
  - Bounding box aspect ratio (posture change indicator)
  - Distance-to-floor estimation (from depth data)
  - Center of mass height
  - Joint vertical velocities (frame-to-frame)
- Build the end-to-end preprocessing pipeline (RGB frames → features saved as .npy files)
- Implement sliding window sequence generation for temporal analysis
- Create train/validation/test split with class balancing
- Run preprocessing on all downloaded URFD sequences

**Deliverables:**
- Feature extraction module producing 177-dimensional vectors
- Preprocessed dataset in .npy format
- Data statistics report (number of samples, class distribution)

---

### Week 4 (Mar 10 - Mar 16): LSTM Model Design and Training

**Objectives:**
- Design and implement the LSTM-based classifier (Section 4.4)
- Train the model and tune hyperparameters
- Analyze initial training results

**Tasks:**
- Implement the Bidirectional LSTM architecture with attention mechanism
- Add batch normalization and dropout for regularization
- Write the training script with:
  - Early stopping (patience = 10 epochs)
  - Learning rate scheduling (ReduceLROnPlateau)
  - Model checkpointing (save best model by validation loss)
- Train on URFD dataset (target: 50 epochs maximum)
- Experiment with hyperparameters:
  - Hidden size: 64 vs 128 vs 256
  - Number of LSTM layers: 1 vs 2 vs 3
  - Sequence length: 20 vs 30 vs 40 frames
  - Learning rate: 0.0001 vs 0.001 vs 0.01
- Log training curves (loss, accuracy per epoch)
- Perform cross-validation to check for overfitting

**Deliverables:**
- Trained LSTM model (best_model.pth)
- Training curves plot (loss and accuracy)
- Hyperparameter comparison table

---

### Week 5 (Mar 17 - Mar 23): Evaluation, Inference, and Extended Testing

**Objectives:**
- Full evaluation using all metrics defined in Section 4.5
- Build video inference pipeline with visualization
- Test on additional data for robustness

**Tasks:**
- Run evaluation on held-out test set and compute:
  - Accuracy, Sensitivity (Recall), Specificity
  - Precision, F1-Score, False Alarm Rate
  - AUC-ROC
- Generate confusion matrix and ROC curve plots
- Build the inference script for processing video files:
  - Skeleton overlay on RGB frames
  - Depth map visualization (side-by-side)
  - Fall detection alerts with confidence scores
  - Output annotated video file
- Download and test with additional URFD sequences (full 30 falls + 40 ADL)
- Optionally download NTU RGB+D subset for supplementary evaluation
- Compare results with existing methods from literature

**Deliverables:**
- Evaluation metrics report
- Confusion matrix and ROC curve plots
- Demo video showing fall detection in action
- Comparison table with literature results

---

### Week 6 (Mar 24 - Mar 30): Documentation, Final Report, and Presentation

**Objectives:**
- Write final documentation and project report
- Prepare presentation for professor
- Clean up repository for submission

**Tasks:**
- Write the Results and Discussion chapter:
  - Present all evaluation metrics in tables
  - Include confusion matrix and ROC curve figures
  - Discuss training behavior (convergence, early stopping)
  - Analyze failure cases if any
- Write the Conclusion chapter:
  - Summarize key findings
  - Discuss limitations (dataset size, generalization)
  - Suggest future work (real-time deployment, multi-camera support)
- Clean up code:
  - Add comments and docstrings where needed
  - Remove any debug or temporary code
  - Update README with final instructions
- Prepare presentation slides:
  - Problem statement and motivation
  - System architecture diagram
  - Key results and plots
  - Demo video
- Final commit and push to GitHub

**Deliverables:**
- Complete project report
- Presentation slides
- Clean, well-documented GitHub repository
- Demo video

---

### Summary Timeline

| Week | Focus Area | Key Output |
|------|-----------|------------|
| Week 1 | Setup + Data | Environment, dataset, repo |
| Week 2 | Depth + Pose | MiDaS and pose modules |
| Week 3 | Features + Pipeline | 177-dim feature vector, preprocessed data |
| Week 4 | Model + Training | Trained LSTM, training curves |
| Week 5 | Evaluation + Demo | Metrics, confusion matrix, demo video |
| Week 6 | Report + Presentation | Final documentation, slides |
