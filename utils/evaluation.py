import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetectionEvaluator:
    """
    Comprehensive evaluator for anomaly detection models.
    Provides standard evaluation metrics and visualizations.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.metrics = {}
        self.predictions = {}
        
    def calculate_metrics(self, y_true, y_pred, y_scores=None):
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true (np.array): True labels
            y_pred (np.array): Predicted labels
            y_scores (np.array, optional): Prediction scores/probabilities
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        # Basic classification metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Calculate additional metrics from confusion matrix
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        metrics['true_positives'] = tp
        
        # Specificity and sensitivity
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # FPR and TPR
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # AUC metrics if scores are provided
        if y_scores is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
                metrics['pr_auc'] = average_precision_score(y_true, y_scores)
            except ValueError:
                metrics['roc_auc'] = 0.0
                metrics['pr_auc'] = 0.0
        
        self.metrics = metrics
        return metrics
    
    def evaluate_model(self, model, X_test, y_test, threshold=None):
        """
        Evaluate a trained anomaly detection model.
        
        Args:
            model: Trained anomaly detection model
            X_test (pd.DataFrame): Test features
            y_test (np.array): Test labels
            threshold (float, optional): Custom threshold for prediction
            
        Returns:
            dict: Evaluation results
        """
        # Get predictions
        if threshold is not None:
            # Use custom threshold
            scores = model.predict_proba(X_test)
            y_pred = (scores > threshold).astype(int)
        else:
            # Use model's default threshold
            y_pred = model.predict(X_test)
            scores = model.predict_proba(X_test)
        
        # Store predictions
        self.predictions = {
            'y_true': y_test,
            'y_pred': y_pred,
            'y_scores': scores
        }
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, scores)
        
        return metrics
    
    def plot_confusion_matrix(self, cm=None, title="Confusion Matrix"):
        """
        Plot confusion matrix.
        
        Args:
            cm (np.array, optional): Confusion matrix
            title (str): Plot title
        """
        if cm is None:
            cm = self.metrics.get('confusion_matrix')
            if cm is None:
                raise ValueError("No confusion matrix available")
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        return plt.gcf()
    
    def plot_roc_curve(self, y_true=None, y_scores=None, title="ROC Curve"):
        """
        Plot ROC curve.
        
        Args:
            y_true (np.array, optional): True labels
            y_scores (np.array, optional): Prediction scores
            title (str): Plot title
        """
        if y_true is None or y_scores is None:
            y_true = self.predictions.get('y_true')
            y_scores = self.predictions.get('y_scores')
            if y_true is None or y_scores is None:
                raise ValueError("No prediction data available")
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_precision_recall_curve(self, y_true=None, y_scores=None, title="Precision-Recall Curve"):
        """
        Plot precision-recall curve.
        
        Args:
            y_true (np.array, optional): True labels
            y_scores (np.array, optional): Prediction scores
            title (str): Plot title
        """
        if y_true is None or y_scores is None:
            y_true = self.predictions.get('y_true')
            y_scores = self.predictions.get('y_scores')
            if y_true is None or y_scores is None:
                raise ValueError("No prediction data available")
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        auc = average_precision_score(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_reconstruction_error_distribution(self, normal_errors, anomaly_errors=None, title="Reconstruction Error Distribution"):
        """
        Plot distribution of reconstruction errors.
        
        Args:
            normal_errors (np.array): Reconstruction errors for normal samples
            anomaly_errors (np.array, optional): Reconstruction errors for anomalous samples
            title (str): Plot title
        """
        plt.figure(figsize=(10, 6))
        
        # Plot normal errors
        plt.hist(normal_errors, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
        
        # Plot anomaly errors if provided
        if anomaly_errors is not None:
            plt.hist(anomaly_errors, bins=50, alpha=0.7, label='Anomaly', color='red', density=True)
        
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Density')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_metrics_comparison(self, metrics_dict, title="Metrics Comparison"):
        """
        Plot comparison of different metrics.
        
        Args:
            metrics_dict (dict): Dictionary of metric names and values
            title (str): Plot title
        """
        metrics_names = list(metrics_dict.keys())
        metrics_values = list(metrics_dict.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics_names, metrics_values, color='skyblue', alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.title(title)
        plt.ylabel('Score')
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    def generate_report(self, model_name="Anomaly Detection Model"):
        """
        Generate a comprehensive evaluation report.
        
        Returns:
            str: Formatted evaluation report
        """
        if not self.metrics:
            return "No evaluation metrics available. Run evaluate_model() first."
        
        report = f"""
# {model_name} - Evaluation Report

## Classification Metrics
- **Accuracy**: {self.metrics['accuracy']:.4f}
- **Precision**: {self.metrics['precision']:.4f}
- **Recall**: {self.metrics['recall']:.4f}
- **F1-Score**: {self.metrics['f1_score']:.4f}

## Detailed Metrics
- **Specificity**: {self.metrics['specificity']:.4f}
- **Sensitivity**: {self.metrics['sensitivity']:.4f}
- **False Positive Rate**: {self.metrics['false_positive_rate']:.4f}
- **True Positive Rate**: {self.metrics['true_positive_rate']:.4f}

## Confusion Matrix
- **True Negatives**: {self.metrics['true_negatives']}
- **False Positives**: {self.metrics['false_positives']}
- **False Negatives**: {self.metrics['false_negatives']}
- **True Positives**: {self.metrics['true_positives']}
"""
        
        # Add AUC metrics if available
        if 'roc_auc' in self.metrics:
            report += f"""
## AUC Metrics
- **ROC AUC**: {self.metrics['roc_auc']:.4f}
- **PR AUC**: {self.metrics['pr_auc']:.4f}
"""
        
        return report
    
    def get_best_threshold(self, y_true, y_scores, metric='f1'):
        """
        Find the best threshold based on a specific metric.
        
        Args:
            y_true (np.array): True labels
            y_scores (np.array): Prediction scores
            metric (str): Metric to optimize ('f1', 'precision', 'recall')
            
        Returns:
            tuple: (best_threshold, best_score)
        """
        thresholds = np.linspace(0, 1, 100)
        scores = []
        
        for threshold in thresholds:
            y_pred = (y_scores > threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            else:
                raise ValueError("metric must be 'f1', 'precision', or 'recall'")
            
            scores.append(score)
        
        best_idx = np.argmax(scores)
        best_threshold = thresholds[best_idx]
        best_score = scores[best_idx]
        
        return best_threshold, best_score 