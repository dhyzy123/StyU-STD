## StyU-STD

This repository contains the core implementation of 'StyU-STD'.

### Style-Diverse Sample Generation from Unlabeled Data

#### 🔹 UnlabeledData.py

UnlabeledSTD: Generates positive training samples by randomly clipping segments from unlabeled speech files.

STDDataConstructor--forword method: Implements Speaker Transformation (ST) and Speaking State Emulation (SSE). ST can be implemented fixed for q, or it can be randomized to choose whether it is implemented for q or s. SSE techniques include: RR: random resampling to simulate rhythm; CP: random cropping to simulate omissions; NS: adding random noise to simulate noise. If config is true, implement, otherwise skip. 

STDDataConstructor--sys method: Manages random selection of negative samples and applies speaker transformation. If the ST parameter is enabled (True), it performs both random negative sample selection and speaker transformation; if disabled (False), it only selects random negative samples.

#### 🔹 data_gen.py

FreeVC (ST): Utilizes the FreeVC framework, in which parameters are initialized with pre-trained weights and kept frozen, enabling robust and flexible speaker transformation, which in turn facilitates more effective training of the subsequent detection network.

### Style Suppressed Convolutional Network

#### 🔹 STD_Model.py

STD_Model: Defines the core detection model designed for training on generated sample data. The model emphasizes content detection while suppressing style variations.

MatrixCosineSimilarity: Multiplys feat_s by feat_q or feat_q by feat_s is feasible.
