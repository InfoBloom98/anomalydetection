import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
import joblib
import os

class Autoencoder:
    """
    Autoencoder model for anomaly detection in network traffic.
    Learns the distribution of normal traffic and detects anomalies
    based on reconstruction error.
    """
    
    def __init__(self, input_dim, encoding_dim=32, hidden_dims=[64, 32]):
        """
        Initialize the autoencoder.
        
        Args:
            input_dim (int): Number of input features
            encoding_dim (int): Dimension of the latent space
            hidden_dims (list): List of hidden layer dimensions for encoder/decoder
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.threshold = None
        self.is_trained = False
        
    def build_model(self):
        """Build the autoencoder architecture."""
        # Input layer
        input_layer = layers.Input(shape=(self.input_dim,))
        
        # Encoder
        encoded = input_layer
        for dim in self.hidden_dims:
            encoded = layers.Dense(dim, activation='relu')(encoded)
            encoded = layers.Dropout(0.2)(encoded)
        
        # Latent representation
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='encoded')(encoded)
        
        # Decoder
        decoded = encoded
        for dim in reversed(self.hidden_dims):
            decoded = layers.Dense(dim, activation='relu')(decoded)
            decoded = layers.Dropout(0.2)(decoded)
        
        # Output layer
        decoded = layers.Dense(self.input_dim, activation='sigmoid', name='decoded')(decoded)
        
        # Create models
        self.encoder = Model(input_layer, encoded, name='encoder')
        self.decoder = Model(encoded, decoded, name='decoder')
        self.autoencoder = Model(input_layer, decoded, name='autoencoder')
        
        # Compile the model
        self.autoencoder.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return self.autoencoder
    
    def train(self, X_train, X_val=None, epochs=100, batch_size=32, verbose=1):
        """
        Train the autoencoder on normal data.
        
        Args:
            X_train (np.array): Training data (normal traffic only)
            X_val (np.array): Validation data
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            verbose (int): Verbosity level
        """
        if self.autoencoder is None:
            self.build_model()
        
        # Training callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train the model
        if X_val is not None:
            history = self.autoencoder.fit(
                X_train, X_train,
                validation_data=(X_val, X_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=verbose
            )
        else:
            history = self.autoencoder.fit(
                X_train, X_train,
                validation_split=0.2,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=verbose
            )
        
        # Calculate reconstruction error threshold
        self._calculate_threshold(X_train)
        self.is_trained = True
        
        return history
    
    def _calculate_threshold(self, X, percentile=95):
        """
        Calculate reconstruction error threshold for anomaly detection.
        
        Args:
            X (np.array): Normal training data
            percentile (float): Percentile for threshold calculation
        """
        # Get reconstruction errors for normal data
        reconstructed = self.autoencoder.predict(X)
        mse = np.mean(np.square(X - reconstructed), axis=1)
        
        # Set threshold at specified percentile
        self.threshold = np.percentile(mse, percentile)
        
        return self.threshold
    
    def predict(self, X):
        """
        Predict anomalies based on reconstruction error.
        
        Args:
            X (np.array): Input data
            
        Returns:
            np.array: Binary predictions (0: normal, 1: anomaly)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Get reconstruction errors
        reconstructed = self.autoencoder.predict(X)
        mse = np.mean(np.square(X - reconstructed), axis=1)
        
        # Classify as anomaly if error exceeds threshold
        predictions = (mse > self.threshold).astype(int)
        
        return predictions
    
    def get_reconstruction_error(self, X):
        """
        Get reconstruction errors for input data.
        
        Args:
            X (np.array): Input data
            
        Returns:
            np.array: Reconstruction errors
        """
        reconstructed = self.autoencoder.predict(X)
        mse = np.mean(np.square(X - reconstructed), axis=1)
        return mse
    
    def encode(self, X):
        """
        Encode input data to latent representation.
        
        Args:
            X (np.array): Input data
            
        Returns:
            np.array: Encoded representation
        """
        return self.encoder.predict(X)
    
    def decode(self, encoded_data):
        """
        Decode latent representation back to original space.
        
        Args:
            encoded_data (np.array): Encoded data
            
        Returns:
            np.array: Decoded data
        """
        return self.decoder.predict(encoded_data)
    
    def save_model(self, filepath):
        """
        Save the trained model and parameters.
        
        Args:
            filepath (str): Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model weights
        self.autoencoder.save_weights(f"{filepath}_weights.h5")
        
        # Save model parameters
        model_params = {
            'input_dim': self.input_dim,
            'encoding_dim': self.encoding_dim,
            'hidden_dims': self.hidden_dims,
            'threshold': self.threshold,
            'is_trained': self.is_trained
        }
        joblib.dump(model_params, f"{filepath}_params.pkl")
    
    def load_model(self, filepath):
        """
        Load a trained model and parameters.
        
        Args:
            filepath (str): Path to the saved model
        """
        # Load model parameters
        model_params = joblib.load(f"{filepath}_params.pkl")
        
        # Set parameters
        self.input_dim = model_params['input_dim']
        self.encoding_dim = model_params['encoding_dim']
        self.hidden_dims = model_params['hidden_dims']
        self.threshold = model_params['threshold']
        self.is_trained = model_params['is_trained']
        
        # Build and load model weights
        self.build_model()
        self.autoencoder.load_weights(f"{filepath}_weights.h5")
    
    def get_model_summary(self):
        """Get a summary of the model architecture."""
        if self.autoencoder is None:
            self.build_model()
        
        summary = []
        self.autoencoder.summary(print_fn=lambda x: summary.append(x))
        return '\n'.join(summary) 