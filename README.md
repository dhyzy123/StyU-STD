# 🧠 StyU-STD

**Official Implementation of _StyU-STD_ (Accepted at SMC 2025)**  
A framework for **style-diverse sample generation from unlabeled speech data** and a **style-suppressed detection network**.

---

## 🎯 Overview: Style-Diverse Sample Generation from Unlabeled Data

StyU-STD enables training a speech detection model **without any labeled data**, by automatically generating style-varied positive and negative samples. It consists of:

- **Unlabeled Sample Constructor**  
  Dynamically generates training samples with varying speaker styles and speaking states.

- **Style-Suppressed Detection Network**  
  Learns to recognize speaker-independent speech content while suppressing style variations.

---

## 📁 Module Descriptions

### 🔹 `UnlabeledData.py`

#### `UnlabeledSTD`
Generates **positive training samples** by randomly clipping segments from unlabeled speech files.

#### `STDDataConstructor.forward()`

Implements the two key augmentation techniques:

- **Speaker Transformation (ST)**:
  - Transforms speaker identity to diversify style.
  - Can be fixed to `q`, or randomly applied to `q` or `s`.
  - _(Only the fixed-`q` version is used in the paper.)_

- **Speaking State Emulation (SSE)**:
  - `RR` – Random Resampling: simulates rhythm changes.
  - `CP` – Cropping: simulates speech omissions.
  - `NS` – Noise: adds random noise to simulate environmental disturbance.

All techniques are conditionally applied based on configuration flags.

#### `STDDataConstructor.sys()`

Handles **negative sample generation**:

- If `ST=True`:
  - Selects a random negative sample.
  - Applies speaker transformation to increase variation.
- If `ST=False`:
  - Only performs negative sampling without transformation.

---

### 🔹 `data_gen.py`

#### `FreeVC (Speaker Transformation)`

Utilizes **FreeVC**, a high-quality voice conversion model:

- Pre-trained and **weights frozen** during use.
- Enables **robust and flexible speaker transformation**.
- Facilitates better training of the downstream detection network.

---

## 🧩 Detection Model

### 🔹 `STD_Model.py`

#### `STD_Model`

The **core detection network** designed to:

- Detect speaker-invariant **speech content**.
- Suppress variations introduced by speaker identity and speaking states.

Trains entirely on synthesized positive and negative samples.

#### `MatrixCosineSimilarity`

Computes cosine similarity between features:

- Commonly uses `feat_q × feat_s`, as shown in the paper.
- Alternatively, `feat_s × feat_q` is also valid and yields similar results.
