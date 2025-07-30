#!/usr/bin/env python3
"""
Training script for the autoencoder-based anomaly detection system.
This script demonstrates the complete training and evaluation pipeline.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from models.anomaly_detector import AnomalyDetector
from utils.data_preprocessing import NetworkDataPreprocessor
from utils.evaluation import AnomalyDetectionEvaluator
from utils.visualization import AnomalyDetectionVisualizer
from data.sample_data import NetworkDataGenerator

def main():
    """Main training function."""
    print("🚀 Starting Autoencoder Anomaly Detection Training")
    print("=" * 60)
    
    # Step 1: Generate sample data
    print("\n📊 Generating sample network traffic data...")
    data_generator = NetworkDataGenerator(random_state=42)
    X, y = data_generator.generate_complex_network_data(
        n_samples=15000, 
        n_features=25, 
        anomaly_ratio=0.15
    )
    
    print(f"Generated {len(X)} samples with {len(X.columns)} features")
    print(f"Anomaly ratio: {np.mean(y):.3f}")
    
    # Step 2: Preprocess data
    print("\n🔧 Preprocessing data...")
    preprocessor = NetworkDataPreprocessor(scaler_type='standard')
    X_processed, y = preprocessor.fit_transform(X, y)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_processed, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Step 3: Train the model
    print("\n🤖 Training autoencoder model...")
    detector = AnomalyDetector(
        input_dim=X_train.shape[1],
        encoding_dim=16,
        hidden_dims=[64, 32]
    )
    
    # Train on normal data only
    normal_mask = y_train == 0
    X_train_normal = X_train[normal_mask]
    
    history = detector.fit(
        X_train_normal,
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    print("✅ Model training completed!")
    
    # Step 4: Evaluate the model
    print("\n📈 Evaluating model performance...")
    evaluator = AnomalyDetectionEvaluator()
    
    # Evaluate on test set
    metrics = evaluator.evaluate_model(detector, X_test, y_test)
    
    # Print results
    print("\n📊 Model Performance Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"PR AUC: {metrics['pr_auc']:.4f}")
    
    # Step 5: Create visualizations
    print("\n📊 Creating visualizations...")
    visualizer = AnomalyDetectionVisualizer()
    
    # Training history
    fig_history = visualizer.plot_training_history(history)
    fig_history.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.close(fig_history)
    
    # Confusion matrix
    fig_cm = evaluator.plot_confusion_matrix()
    fig_cm.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close(fig_cm)
    
    # ROC curve
    y_true = evaluator.predictions['y_true']
    y_scores = evaluator.predictions['y_scores']
    fig_roc = evaluator.plot_roc_curve(y_true, y_scores)
    fig_roc.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close(fig_roc)
    
    # Precision-Recall curve
    fig_pr = evaluator.plot_precision_recall_curve(y_true, y_scores)
    fig_pr.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close(fig_pr)
    
    # Reconstruction error distribution
    normal_errors = detector.predict_proba(X_test[y_test == 0])
    anomaly_errors = detector.predict_proba(X_test[y_test == 1])
    fig_error = evaluator.plot_reconstruction_error_distribution(normal_errors, anomaly_errors)
    fig_error.savefig('reconstruction_error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close(fig_error)
    
    # Metrics comparison
    relevant_metrics = {
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1-Score': metrics['f1_score'],
        'ROC AUC': metrics['roc_auc']
    }
    fig_metrics = evaluator.plot_metrics_comparison(relevant_metrics)
    fig_metrics.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig_metrics)
    
    print("✅ Visualizations saved!")
    
    # Step 6: Save the model
    print("\n💾 Saving trained model...")
    detector.save_model('models/trained_anomaly_detector')
    print("✅ Model saved to 'models/trained_anomaly_detector'")
    
    # Step 7: Generate comprehensive report
    print("\n📋 Generating evaluation report...")
    report = evaluator.generate_report("Autoencoder Anomaly Detection Model")
    
    with open('evaluation_report.md', 'w') as f:
        f.write(report)
    
    print("✅ Evaluation report saved to 'evaluation_report.md'")
    
    # Step 8: Model information
    print("\n🔍 Model Information:")
    model_info = detector.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    print("\n🎉 Training and evaluation completed successfully!")
    print("=" * 60)
    
    return detector, evaluator, visualizer

def demonstrate_prediction(detector, X_sample, y_sample):
    """
    Demonstrate prediction on sample data.
    
    Args:
        detector: Trained anomaly detector
        X_sample: Sample features
        y_sample: Sample labels
    """
    print("\n🔮 Demonstration: Making predictions on sample data...")
    
    # Make predictions
    predictions = detector.predict(X_sample)
    scores = detector.predict_proba(X_sample)
    
    # Show results
    print(f"Sample size: {len(X_sample)}")
    print(f"True anomalies: {np.sum(y_sample)}")
    print(f"Predicted anomalies: {np.sum(predictions)}")
    print(f"Average reconstruction error: {np.mean(scores):.4f}")
    
    # Show some examples
    print("\n📋 Sample predictions:")
    for i in range(min(5, len(X_sample))):
        status = "ANOMALY" if predictions[i] == 1 else "NORMAL"
        true_status = "ANOMALY" if y_sample[i] == 1 else "NORMAL"
        error = scores[i]
        print(f"  Sample {i+1}: Predicted={status}, Actual={true_status}, Error={error:.4f}")

if __name__ == "__main__":
    # Run the main training
    detector, evaluator, visualizer = main()
    
    # Demonstrate prediction on a small sample
    print("\n" + "=" * 60)
    print("DEMONSTRATION")
    print("=" * 60)
    
    # Generate a small test sample
    data_generator = NetworkDataGenerator(random_state=123)
    X_demo, y_demo = data_generator.generate_basic_network_data(
        n_samples=100, 
        n_features=20, 
        anomaly_ratio=0.2
    )
    
    # Preprocess the demo data
    preprocessor = NetworkDataPreprocessor(scaler_type='standard')
    X_demo_processed, y_demo = preprocessor.fit_transform(X_demo, y_demo)
    
    # Demonstrate prediction
    demonstrate_prediction(detector, X_demo_processed, y_demo)
    
    print("\n🎯 Autoencoder Anomaly Detection System Ready!")
    print("You can now use the trained model for real-time anomaly detection.") 