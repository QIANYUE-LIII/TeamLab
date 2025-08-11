# TeamLab2025_Q3 - Deepfake Audio Detection

## Project Intro
The current project aims at developing a pipeline countermeasure (CM) for deepfake audio spoofing attacks based ASVspoof2019 LA dataset. Two sets of features are employed in the front-end feature extraction, namely explainable/prosody features (pitch, HNR, jitter, shimmer) and non-explainable features (MFCCs). For the back-end classifiers, LSTM is employed for pitch and HNR, CNN for MFCCs, FC for jitter and shimmer. Different branches are combined through late fusion as an ensemble model. 

The scripts used in the project are outlined as below

## Evaluation Metric
- *eval_metrics.py*: for EER calculation. Provided by ASVspoof2019 challenge.

## Feature extraction
- *prosody_features.ipynb*: prosody feature extraction
- *MFCC_extraction.ipynb*: MFCC feature extraction

## Models
- *baseline_cnn.ipynb*: baseline CNN development
- *training_allfeature.ipynb*: training script for LSTM-FC, CNN, and EM classifiers
- *models.py*: seperated classifier file used for testing
- *testing.ipynb*
- *training_allfeature_v2.ipynb*: upgraded version of the training script. Allows for each feature and each branch classifier to be trained seperately.
- *models_v2.py*: see above
- *testing_v2.ipynb*: see above

## Analysis
- *attack_analysis.ipynb*: table and figure plotting for testing results on different attack types
- *grad_cam_visualization.ipynb*: GradCAM analysis for CNN classifier

## Miscellaneous
- *MFCC_load_feature_labels.ipynb*: mapping MFCC features to their corresponding golden labels
- *miscellaneous.ipynb*: format conversion, compression, etc.
- *scaling.ipynb*: Min-Max scaling for all the features