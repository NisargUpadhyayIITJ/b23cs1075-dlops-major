# DLOps Major Exam 2026

**Name:** Nisarg Upadhyay  
**Roll No:** B23CS1075

---

## Question 1: NLP Translation Task (5 Marks)

**Model:** Helsinki-NLP/opus-mt-bn-en

**Output generated for first statement:** "I have a test today."

**BLEU Score:** 38.22

---

## Question 2: CityScape Image Segmentation (10 Marks)

**Model:** UNet (23 segmentation classes)  
**Training:** 25 epochs, Adam optimizer (lr=1e-3), CrossEntropyLoss  
**Image Size:** 128x96, Dataset Split: 80-20 (seed=42)

**Test Set Results:**

Question2: mIOU: 0.6283 and mDICE: 0.6864

---

## Question 4: ECAPA-TDNN Model Optimization and Quantization (10 Marks)

**Model:** ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb)  
**Dataset:** s3prl/superb (SI split; Val for finetuning, Eval for test)

### Task 1: Baseline Inference and Basic Profiling
- **Baseline Accuracy:** 0.9697 (96.97%)
- **Baseline GFLOPs:** 9.4185

### Task 2: Post-Training Quantization (INT8)
- **PTQ GFLOPs:** 2.3546
- **GFLOPs Impact:** 7.0639 reduction compared to baseline

### Task 3: PTQ Accuracy
- **PTQ Accuracy:** 0.5745 (decreased from baseline)

### Task 4: QAT with Optuna (5 trials)
- **Best Hyperparameters:** learning_rate=2.277e-06, weight_decay=0.003368, num_epochs=2
- **Best QAT Accuracy:** 0.6647 (66.47%)
- **QAT GFLOPs:** 2.3546

### Task 5: Final Analysis
- **Total Accuracy Diff (QAT vs Baseline):** -0.3050 (absolute)
- **GFLOPs Saved:** 7.0639

---
