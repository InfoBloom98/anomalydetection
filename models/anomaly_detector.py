import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .autoencoder import Autoencoder
import joblib
import os

class AnomalyDetector:
    """
    High-level wrapper for autoencoder-based anomaly detection.
    Handles data preprocessing, model training, and prediction.
    """
    
    def __init__(self, input_dim=None, encoding_dim=32, hidden_dims=[64, 32]):
        """
        Initialize the anomaly detector.
        
        Args:
            input_dim (int): Number of input features (will be inferred from data if None)
            encoding_dim (int): Dimension of the latent space
            hidden_dims (list): List of hidden layer dimensions
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims
        self.autoencoder = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X, epochs=100, batch_size=32, verbose=1):
        """
        Fit the anomaly detector on normal data.
        
        Args:
            X (np.array or pd.DataFrame): Training data (normal traffic only)
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            verbose (int): Verbosity level
        """
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Determine input dimension if not specified
        if self.input_dim is None:
            self.input_dim = X.shape[1]
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize and train autoencoder
        self.autoencoder = Autoencoder(
            input_dim=self.input_dim,
            encoding_dim=self.encoding_dim,
            hidden_dims=self.hidden_dims
        )
        
        # Train the model
        history = self.autoencoder.train(
            X_scaled, 
            epochs=epochs, 
            batch_size=batch_size, 
            verbose=verbose
        )
        
        self.is_fitted = True
        return history
    
    def predict(self, X):
        """
        Predict anomalies in the data.
        
        Args:
            X (np.array or pd.DataFrame): Input data
            
        Returns:
            np.array: Binary predictions (0: normal, 1: anomaly)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Scale the data
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.autoencoder.predict(X_scaled)
        
        return predictions
    
    def predict_proba(self, X):
        """
        Get reconstruction error scores (higher = more anomalous).
        
        Args:
            X (np.array or pd.DataFrame): Input data
            
        Returns:
            np.array: Reconstruction error scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Scale the data
        X_scaled = self.scaler.transform(X)
        
        # Get reconstruction errors
        errors = self.autoencoder.get_reconstruction_error(X_scaled)
        
        return errors
    
    def get_threshold(self):
        """
        Get the current anomaly detection threshold.
        
        Returns:
            float: Current threshold value
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting threshold")
        
        return self.autoencoder.threshold
    
    def set_threshold(self, threshold):
        """
        Set a custom anomaly detection threshold.
        
        Args:
            threshold (float): New threshold value
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before setting threshold")
        
        self.autoencoder.threshold = threshold
    
    def get_model_info(self):
        """
        Get information about the trained model.
        
        Returns:
            dict: Model information
        """
        if not self.is_fitted:
            return {"status": "Not fitted"}
        
        return {
            "input_dim": self.input_dim,
            "encoding_dim": self.encoding_dim,
            "hidden_dims": self.hidden_dims,
            "threshold": self.autoencoder.threshold,
            "is_trained": self.autoencoder.is_trained,
            "scaler_fitted": self.scaler is not None
        }
    
    def save_model(self, filepath):
        """
        Save the trained model and scaler.
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save autoencoder
        self.autoencoder.save_model(f"{filepath}_autoencoder")
        
        # Save scaler
        joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
        
        # Save detector parameters
        detector_params = {
            'input_dim': self.input_dim,
            'encoding_dim': self.encoding_dim,
            'hidden_dims': self.hidden_dims,
            'is_fitted': self.is_fitted
        }
        joblib.dump(detector_params, f"{filepath}_detector_params.pkl")
    
    def load_model(self, filepath):
        """
        Load a trained model and scaler.
        
        Args:
            filepath (str): Path to the saved model
        """
        # Load detector parameters
        detector_params = joblib.load(f"{filepath}_detector_params.pkl")
        
        # Set parameters
        self.input_dim = detector_params['input_dim']
        self.encoding_dim = detector_params['encoding_dim']
        self.hidden_dims = detector_params['hidden_dims']
        self.is_fitted = detector_params['is_fitted']
        
        # Load autoencoder
        self.autoencoder = Autoencoder(
            input_dim=self.input_dim,
            encoding_dim=self.encoding_dim,
            hidden_dims=self.hidden_dims
        )
        self.autoencoder.load_model(f"{filepath}_autoencoder")
        
        # Load scaler
        self.scaler = joblib.load(f"{filepath}_scaler.pkl")
    
    def get_reconstruction_quality(self, X):
        """
        Get reconstruction quality metrics for input data.
        
        Args:
            X (np.array or pd.DataFrame): Input data
            
        Returns:
            dict: Reconstruction quality metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting reconstruction quality")
        
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Scale the data
        X_scaled = self.scaler.transform(X)
        
        # Get reconstruction
        reconstructed = self.autoencoder.autoencoder.predict(X_scaled)
        
        # Calculate metrics
        mse = np.mean(np.square(X_scaled - reconstructed), axis=1)
        mae = np.mean(np.abs(X_scaled - reconstructed), axis=1)
        
        return {
            'mse': mse,
            'mae': mae,
            'mean_mse': np.mean(mse),
            'mean_mae': np.mean(mae),
            'std_mse': np.std(mse),
            'std_mae': np.std(mae)
        } 