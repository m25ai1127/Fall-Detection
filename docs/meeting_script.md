# Professor Meeting Script — Fall Detection Project
## How to Explain Your Project from Start to Finish

---

## PART 1: Project Introduction (2-3 minutes)

**Start like this:**

"Sir, my project is on AI-based fall detection using depth-assisted single-camera vision. The problem I'm trying to solve is — most existing fall detection systems need expensive depth sensors like Kinect or RealSense cameras. My approach replaces that hardware with MiDaS, an AI model that estimates depth from a regular webcam. This makes the system deployable with any existing CCTV camera."

"The system works in three stages: First, MiDaS generates a depth map from the camera frame. Second, MediaPipe detects 33 body joint positions. Third, I combine depth and pose into a 177-dimensional feature vector and feed 30 consecutive frames into a Bidirectional LSTM to classify whether a fall is happening."

---

## PART 2: Why This Approach? (2 minutes)

**If professor asks "Why depth?":**

"A regular 2D camera can only see x and y coordinates. If a person falls toward the camera, the 2D image barely changes — the person just appears slightly bigger. But the depth changes dramatically — from 3 meters to 0.5 meters. Without depth, we would miss falls happening toward or away from the camera."

**If professor asks "Why MiDaS specifically?":**

"MiDaS is the best choice because it runs on CPU without needing a GPU, it's pre-trained by Intel so I don't need to train it separately, and it works with any regular camera. Other depth models like Depth Anything need a GPU, and ZoeDepth is too slow for real-time processing."

**If professor asks "Why not a real depth camera?":**

"The key innovation of my project is eliminating the need for specialised hardware. A Kinect costs 200-400 dollars and is no longer manufactured. My system works with any 500-rupee webcam or existing CCTV camera. This makes it practical for real deployment in hospitals and elderly care homes."

---

## PART 3: Feature Engineering — 177 Dimensions (3-4 minutes)

**Explain like this:**

"For each video frame, I extract 177 numbers that describe the person's body position and movement."

"The first 132 features come from MediaPipe — 33 body joints, each with 4 values: x position, y position, relative z depth, and visibility confidence."

"The next 33 features are the MiDaS depth values sampled at each joint location. This gives real-world depth at every body part."

"Then I compute 3 engineered features: bounding box aspect ratio — which tells if the body shape is tall/narrow (standing) or wide/short (fallen); distance-to-floor — how high the hips are from the ground; and center of mass height — the average height of all joints."

"The final 9 features are vertical velocity — the speed of downward movement for 9 key joints: nose, shoulders, hips, knees, and ankles. This is the most important feature because falls happen fast while sitting down happens slowly."

**If professor asks "Why all 33 joints? Head to ground is enough right?:**

"If we only track head position, we can't distinguish between a person bending down to pick something up and actually falling. The additional joints capture HOW the body moves — arms flailing during a trip, legs going up during a slip, body going stiff during a faint. Each fall type has a different pattern. Also, the LSTM's attention mechanism automatically learns which joints matter most, so including all 33 gives the model maximum information to work with."

**If professor asks "Why only 9 key joints for velocity?":**

"We calculate velocity only for the 9 most relevant joints — nose, shoulders, hips, knees, and ankles. Finger joints or eye/ear landmarks barely move during a fall, so computing their velocity would add noise without useful information. This keeps the feature vector efficient."

---

## PART 4: Why LSTM? (2 minutes)

**Explain like this:**

"Falls are temporal events — they happen over a sequence of frames, not in a single image. An LSTM is specifically designed for sequential data. It looks at 30 consecutive frames — exactly 1 second of video — and understands the pattern over time."

"I use a Bidirectional LSTM, which means it reads the sequence both forward and backward. The forward pass knows what happened before each frame, and the backward pass knows what happens after. Combined, it better understands the full motion pattern."

"I also added an attention mechanism, which lets the model focus on the most important frames — like the exact moment of impact — rather than treating all 30 frames equally."

**If professor asks "Why not Transformer?":**

"Transformers typically need 10,000+ training samples to avoid overfitting. My current dataset has only 1,041 samples, which is too small for a Transformer to generalise well. LSTM works better with limited data. However, in my future work, when I incorporate the NTU RGB+D dataset with 56,000+ samples, switching to a Transformer would be a natural next step."

---

## PART 5: Training & Results (3 minutes)

**Explain like this:**

"I used the URFD dataset — 70 video sequences with 30 falls and 40 normal daily activities like walking, sitting, and picking up objects. Using a sliding window of 30 frames with stride 10, I generated 1,041 training samples."

"The model was trained for 20 epochs with a batch size of 16, learning rate of 0.001, and early stopping with patience of 10. Training took about 20 seconds on CPU."

"On the test set of 157 samples, the model achieved 99.4% accuracy, 97.1% sensitivity — meaning it missed only 1 fall out of 35 — and 100% specificity with zero false alarms. The AUC-ROC is 1.0."

**If professor asks "Why was it 100% initially but 99.4% now?":**

"The initial 100% was on only 21 test samples from 10 sequences — too small to be reliable. The 99.4% is on 157 test samples from the full 70-sequence dataset. This is a much more trustworthy result. The one missed fall was likely a borderline case where the fall motion was very slow."

**If professor asks "What about the 1 missed fall?":**

"Out of 35 fall samples in the test set, the model missed 1. This could be a slow or unusual fall that doesn't match typical fall patterns. Future improvements include adding more diverse fall training data and tuning the classification threshold — lowering it from 0.5 to 0.3 would catch more falls at the cost of slightly more false alarms."

---

## PART 6: Fall Confirmation Window (2 minutes)

**Explain like this:**

"Instead of immediately alerting when a fall is detected, the system has a 3-second confirmation window. When the LSTM confidence first exceeds 50%, the system shows 'Fall Suspected' in orange and starts a 3-second timer."

"If the person stays down for 3 seconds — meaning the confidence remains high — the system confirms the fall with a red alert. But if the person gets back up within those 3 seconds and the confidence drops below 30%, the alert is cancelled."

"This prevents false alarms from stumbles, bending down, or exercises. After a confirmed fall, there is also a 5-second cooldown before the system can trigger another alert, to avoid repeated notifications."

**If professor asks "What about exercise — squats, push-ups?":**

"Currently the model may have false positives for exercise movements because the training data doesn't include exercise videos. The key discriminating features are velocity pattern — falls are sudden one-time events, exercises are rhythmic and repeated — and recovery pattern — falls remain on the ground, exercises return to standing. I plan to address this by adding the NTU RGB+D dataset which contains exercise activities."

---

## PART 7: What About Trajectory? (if asked)

"The LSTM implicitly learns trajectory patterns by analysing 30 consecutive frames of joint positions — it sees how each joint moves over 1 second. The vertical velocity features explicitly capture the speed of trajectory change. As a future enhancement, I could add acceleration features — the change in velocity — to better detect the sudden deceleration when a person hits the ground."

---

## PART 8: Configuration Parameters (if asked in detail)

"All hyperparameters are centralised in config.py. Key settings:
- Sequence length is 30 frames (1 second at 30 FPS)
- Sliding window stride is 10 (creates overlapping training samples)
- LSTM has 2 layers with hidden size 128, bidirectional with dropout 0.3
- Training uses Adam optimiser with learning rate 0.001 and weight decay 0.0001
- Early stopping patience is 10 epochs, LR scheduler reduces by 0.5 after 5 epochs of no improvement
- Classification threshold is 0.5 — adjustable based on deployment scenario"

---

## PART 9: Limitations & Future Work (2 minutes)

**Be honest — professors love this:**

"There are some limitations I have identified. First, the model hasn't been trained on exercise movements, so push-ups or sit-ups could cause false alarms. Second, it runs at about 2-5 FPS on CPU, which limits real-time performance. Third, the dataset is relatively small with only 70 sequences."

"For future work in the remaining 2 weeks, I plan to add the NTU RGB+D dataset for exercise-like activities, implement cross-validation for more robust evaluation, experiment with a Transformer-based architecture, and add acceleration features for better trajectory analysis."

---

## QUICK REFERENCE — If Professor Asks These:

| Question | Short Answer |
|----------|-------------|
| Why depth? | 2D camera misses falls toward/away from camera |
| Why MiDaS? | Runs on CPU, pre-trained, works with any webcam |
| Why 177 features? | 132 pose + 33 depth + 3 body shape + 9 velocity |
| Why all 33 joints? | Different falls use different body parts; model learns which matter |
| Why LSTM not Transformer? | Only 1041 samples — too small for Transformer |
| Why bidirectional? | Reads forward + backward for better context |
| What's attention? | Focuses on important frames (moment of fall) |
| Why 30 frames? | 1 second of video — typical fall duration |
| What's the 3-second wait? | Confirmation window to avoid false alarms |
| What's the 5-second cooldown? | Prevents repeated alerts for same fall |
| Exercise false alarms? | Known limitation — plan to add exercise data |
| Why 99.4% not 100%? | Realistic result on 157 samples (1 borderline miss) |
| Is this original? | Unique combination of MiDaS + 33 joints + BiLSTM + attention |
