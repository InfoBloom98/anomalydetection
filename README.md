# Network Threat Detection with Autoencoder Anomaly Detection

## Overview
This project implements an autoencoder-based anomaly detection system for identifying abnormal network traffic patterns. The system uses deep learning to learn the distribution of normal traffic and detect anomalies based on reconstruction error.

## Features
- **Autoencoder Model**: Deep learning model that learns normal traffic patterns
- **Anomaly Detection**: Identifies abnormal network traffic based on reconstruction error
- **Performance Evaluation**: Comprehensive metrics including precision, recall, F1-score, and AUC
- **Streamlit Web App**: Interactive web interface for model training, evaluation, and prediction
- **Data Visualization**: Real-time plots and charts for model performance analysis

## Project Structure
```
├── models/
│   ├── autoencoder.py          # Autoencoder model implementation
│   └── anomaly_detector.py     # Anomaly detection wrapper
├── utils/
│   ├── data_preprocessing.py   # Data preprocessing utilities
│   ├── evaluation.py           # Evaluation metrics
│   └── visualization.py        # Plotting utilities
├── data/
│   └── sample_data.py          # Sample network traffic data generator
├── app.py                      # Streamlit web application
├── train_model.py              # Model training script
└── requirements.txt            # Python dependencies
```

## Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model
```bash
python train_model.py
```

### Running the Web App
```bash
streamlit run app.py
```

## Model Architecture
The autoencoder consists of:
- **Encoder**: Dense layers that compress input to latent representation
- **Decoder**: Dense layers that reconstruct input from latent representation
- **Anomaly Detection**: Based on reconstruction error threshold

## Evaluation Metrics
- Precision, Recall, F1-Score
- ROC-AUC and PR-AUC
- Confusion Matrix
- Reconstruction Error Distribution

## Web App Features
- Model training with custom parameters
- Real-time performance visualization
- Anomaly detection on new data
- Model export/import functionality 