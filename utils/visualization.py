import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetectionVisualizer:
    """
    Visualization utilities for anomaly detection results.
    Provides both static matplotlib and interactive plotly plots.
    """
    
    def __init__(self):
        """Initialize the visualizer."""
        # Set style for matplotlib
        plt.style.use('default')
        sns.set_palette("husl")
        
    def plot_training_history(self, history, figsize=(12, 4)):
        """
        Plot training history (loss and metrics).
        
        Args:
            history: Keras training history
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Training history plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot loss
        ax1.plot(history.history['loss'], label='Training Loss', color='blue')
        if 'val_loss' in history.history:
            ax1.plot(history.history['val_loss'], label='Validation Loss', color='red')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot metrics
        if 'mae' in history.history:
            ax2.plot(history.history['mae'], label='Training MAE', color='blue')
            if 'val_mae' in history.history:
                ax2.plot(history.history['val_mae'], label='Validation MAE', color='red')
            ax2.set_title('Model MAE')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_interactive_training_history(self, history):
        """
        Create interactive training history plot using plotly.
        
        Args:
            history: Keras training history
            
        Returns:
            plotly.graph_objects.Figure: Interactive training history plot
        """
        epochs = list(range(1, len(history.history['loss']) + 1))
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Model Loss', 'Model MAE'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add loss traces
        fig.add_trace(
            go.Scatter(x=epochs, y=history.history['loss'], 
                      mode='lines', name='Training Loss', line=dict(color='blue')),
            row=1, col=1
        )
        
        if 'val_loss' in history.history:
            fig.add_trace(
                go.Scatter(x=epochs, y=history.history['val_loss'], 
                          mode='lines', name='Validation Loss', line=dict(color='red')),
                row=1, col=1
            )
        
        # Add MAE traces
        if 'mae' in history.history:
            fig.add_trace(
                go.Scatter(x=epochs, y=history.history['mae'], 
                          mode='lines', name='Training MAE', line=dict(color='blue')),
                row=1, col=2
            )
            
            if 'val_mae' in history.history:
                fig.add_trace(
                    go.Scatter(x=epochs, y=history.history['val_mae'], 
                              mode='lines', name='Validation MAE', line=dict(color='red')),
                    row=1, col=2
                )
        
        fig.update_layout(
            title="Training History",
            showlegend=True,
            height=400
        )
        
        return fig
    
    def plot_reconstruction_error_distribution(self, normal_errors, anomaly_errors=None, 
                                            title="Reconstruction Error Distribution"):
        """
        Plot distribution of reconstruction errors.
        
        Args:
            normal_errors (np.array): Reconstruction errors for normal samples
            anomaly_errors (np.array, optional): Reconstruction errors for anomalous samples
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Distribution plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot normal errors
        ax.hist(normal_errors, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
        
        # Plot anomaly errors if provided
        if anomaly_errors is not None:
            ax.hist(anomaly_errors, bins=50, alpha=0.7, label='Anomaly', color='red', density=True)
        
        ax.set_xlabel('Reconstruction Error')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_interactive_error_distribution(self, normal_errors, anomaly_errors=None,
                                         title="Reconstruction Error Distribution"):
        """
        Create interactive reconstruction error distribution plot.
        
        Args:
            normal_errors (np.array): Reconstruction errors for normal samples
            anomaly_errors (np.array, optional): Reconstruction errors for anomalous samples
            title (str): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Interactive distribution plot
        """
        fig = go.Figure()
        
        # Add normal errors histogram
        fig.add_trace(go.Histogram(
            x=normal_errors,
            name='Normal',
            opacity=0.7,
            nbinsx=50,
            marker_color='blue'
        ))
        
        # Add anomaly errors histogram if provided
        if anomaly_errors is not None:
            fig.add_trace(go.Histogram(
                x=anomaly_errors,
                name='Anomaly',
                opacity=0.7,
                nbinsx=50,
                marker_color='red'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Reconstruction Error",
            yaxis_title="Count",
            barmode='overlay',
            height=500
        )
        
        return fig
    
    def plot_confusion_matrix_interactive(self, cm, title="Confusion Matrix"):
        """
        Create interactive confusion matrix plot.
        
        Args:
            cm (np.array): Confusion matrix
            title (str): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Interactive confusion matrix
        """
        labels = ['Normal', 'Anomaly']
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            colorscale='Blues'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            height=500
        )
        
        return fig
    
    def plot_roc_curve_interactive(self, fpr, tpr, auc, title="ROC Curve"):
        """
        Create interactive ROC curve plot.
        
        Args:
            fpr (np.array): False positive rates
            tpr (np.array): True positive rates
            auc (float): AUC score
            title (str): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Interactive ROC curve
        """
        fig = go.Figure()
        
        # Add ROC curve
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC curve (AUC = {auc:.3f})',
            line=dict(color='darkorange', width=2)
        ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random classifier',
            line=dict(color='navy', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1.05]),
            height=500
        )
        
        return fig
    
    def plot_metrics_dashboard(self, metrics_dict, title="Model Performance Metrics"):
        """
        Create interactive metrics dashboard.
        
        Args:
            metrics_dict (dict): Dictionary of metric names and values
            title (str): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Interactive metrics dashboard
        """
        metrics_names = list(metrics_dict.keys())
        metrics_values = list(metrics_dict.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=metrics_names,
                y=metrics_values,
                text=[f'{v:.3f}' for v in metrics_values],
                textposition='auto',
                marker_color='skyblue',
                opacity=0.7
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Metrics",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1.1]),
            height=500
        )
        
        return fig
    
    def plot_feature_importance(self, feature_names, importance_scores, title="Feature Importance"):
        """
        Create interactive feature importance plot.
        
        Args:
            feature_names (list): List of feature names
            importance_scores (list): List of importance scores
            title (str): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Interactive feature importance plot
        """
        # Sort by importance
        sorted_indices = np.argsort(importance_scores)[::-1]
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_scores = [importance_scores[i] for i in sorted_indices]
        
        fig = go.Figure(data=[
            go.Bar(
                x=sorted_scores,
                y=sorted_features,
                orientation='h',
                marker_color='lightcoral',
                opacity=0.7
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=max(400, len(feature_names) * 20)
        )
        
        return fig
    
    def plot_data_distribution(self, X, y, feature_names=None, n_features=5):
        """
        Create interactive data distribution plot.
        
        Args:
            X (pd.DataFrame): Feature data
            y (np.array): Target labels
            feature_names (list, optional): Feature names
            n_features (int): Number of features to plot
            
        Returns:
            plotly.graph_objects.Figure: Interactive data distribution plot
        """
        if feature_names is None:
            feature_names = X.columns.tolist()
        
        # Select top features by variance
        variances = X.var().sort_values(ascending=False)
        top_features = variances.head(n_features).index.tolist()
        
        fig = make_subplots(
            rows=n_features, cols=1,
            subplot_titles=top_features,
            specs=[[{"secondary_y": False}] for _ in range(n_features)]
        )
        
        for i, feature in enumerate(top_features, 1):
            # Normal data
            normal_data = X[y == 0][feature]
            fig.add_trace(
                go.Histogram(
                    x=normal_data,
                    name='Normal' if i == 1 else None,
                    opacity=0.7,
                    marker_color='blue',
                    showlegend=(i == 1)
                ),
                row=i, col=1
            )
            
            # Anomaly data
            anomaly_data = X[y == 1][feature]
            fig.add_trace(
                go.Histogram(
                    x=anomaly_data,
                    name='Anomaly' if i == 1 else None,
                    opacity=0.7,
                    marker_color='red',
                    showlegend=(i == 1)
                ),
                row=i, col=1
            )
        
        fig.update_layout(
            title="Feature Distributions by Class",
            height=200 * n_features,
            barmode='overlay'
        )
        
        return fig
    
    def create_streamlit_plots(self, history=None, metrics=None, predictions=None):
        """
        Create plots for Streamlit app.
        
        Args:
            history: Training history
            metrics: Evaluation metrics
            predictions: Prediction results
            
        Returns:
            dict: Dictionary of plotly figures
        """
        plots = {}
        
        # Training history
        if history is not None:
            plots['training_history'] = self.plot_interactive_training_history(history)
        
        # Metrics dashboard
        if metrics is not None:
            # Filter relevant metrics
            relevant_metrics = {k: v for k, v in metrics.items() 
                              if k in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']}
            plots['metrics_dashboard'] = self.plot_metrics_dashboard(relevant_metrics)
        
        # Confusion matrix
        if predictions is not None and 'confusion_matrix' in metrics:
            plots['confusion_matrix'] = self.plot_confusion_matrix_interactive(
                metrics['confusion_matrix']
            )
        
        # ROC curve
        if predictions is not None and 'roc_auc' in metrics:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(predictions['y_true'], predictions['y_scores'])
            plots['roc_curve'] = self.plot_roc_curve_interactive(
                fpr, tpr, metrics['roc_auc']
            )
        
        return plots 