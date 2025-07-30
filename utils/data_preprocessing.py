import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class NetworkDataPreprocessor:
    """
    Preprocessor for network traffic data.
    Handles feature engineering, scaling, and data splitting.
    """
    
    def __init__(self, scaler_type='standard'):
        """
        Initialize the preprocessor.
        
        Args:
            scaler_type (str): Type of scaler ('standard' or 'minmax')
        """
        self.scaler_type = scaler_type
        self.scaler = None
        self.feature_names = None
        self.is_fitted = False
        
    def create_sample_network_data(self, n_samples=10000, n_features=20, anomaly_ratio=0.1):
        """
        Create synthetic network traffic data for demonstration.
        
        Args:
            n_samples (int): Number of samples to generate
            n_features (int): Number of features
            anomaly_ratio (float): Ratio of anomalous samples
            
        Returns:
            tuple: (X, y) where X is features and y is labels
        """
        np.random.seed(42)
        
        # Generate normal traffic features
        n_normal = int(n_samples * (1 - anomaly_ratio))
        n_anomaly = n_samples - n_normal
        
        # Normal traffic patterns
        normal_data = np.random.normal(0, 1, (n_normal, n_features))
        
        # Anomalous traffic patterns (different distributions)
        anomaly_data = np.random.normal(3, 2, (n_anomaly, n_features))
        
        # Combine data
        X = np.vstack([normal_data, anomaly_data])
        y = np.hstack([np.zeros(n_normal), np.ones(n_anomaly)])
        
        # Shuffle the data
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]
        
        # Create feature names
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        return pd.DataFrame(X, columns=feature_names), y
    
    def engineer_features(self, df):
        """
        Engineer features from raw network data.
        
        Args:
            df (pd.DataFrame): Raw network data
            
        Returns:
            pd.DataFrame: Engineered features
        """
        # Create a copy to avoid modifying original data
        df_eng = df.copy()
        
        # Add statistical features if numeric columns exist
        numeric_cols = df_eng.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            # Add rolling statistics
            for col in numeric_cols[:5]:  # Limit to first 5 columns to avoid too many features
                df_eng[f'{col}_rolling_mean'] = df_eng[col].rolling(window=5, min_periods=1).mean()
                df_eng[f'{col}_rolling_std'] = df_eng[col].rolling(window=5, min_periods=1).std()
            
            # Add lag features
            for col in numeric_cols[:3]:  # Limit to first 3 columns
                df_eng[f'{col}_lag1'] = df_eng[col].shift(1).fillna(method='bfill')
                df_eng[f'{col}_lag2'] = df_eng[col].shift(2).fillna(method='bfill')
            
            # Add interaction features
            if len(numeric_cols) >= 2:
                df_eng['feature_interaction'] = df_eng[numeric_cols[0]] * df_eng[numeric_cols[1]]
        
        # Fill any remaining NaN values
        df_eng = df_eng.fillna(0)
        
        return df_eng
    
    def fit_transform(self, X, y=None):
        """
        Fit the preprocessor and transform the data.
        
        Args:
            X (pd.DataFrame or np.array): Input features
            y (np.array, optional): Target labels
            
        Returns:
            tuple: (X_transformed, y) if y is provided, else X_transformed
        """
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        # Engineer features
        X_eng = self.engineer_features(X)
        
        # Store feature names
        self.feature_names = X_eng.columns.tolist()
        
        # Initialize and fit scaler
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'minmax'")
        
        # Fit and transform
        X_scaled = self.scaler.fit_transform(X_eng)
        X_transformed = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        self.is_fitted = True
        
        if y is not None:
            return X_transformed, y
        else:
            return X_transformed
    
    def transform(self, X):
        """
        Transform new data using fitted preprocessor.
        
        Args:
            X (pd.DataFrame or np.array): Input features
            
        Returns:
            pd.DataFrame: Transformed features
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming")
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        # Engineer features
        X_eng = self.engineer_features(X)
        
        # Transform using fitted scaler
        X_scaled = self.scaler.transform(X_eng)
        X_transformed = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        return X_transformed
    
    def split_data(self, X, y, test_size=0.2, val_size=0.2, random_state=42):
        """
        Split data into train, validation, and test sets.
        
        Args:
            X (pd.DataFrame): Features
            y (np.array): Labels
            test_size (float): Proportion for test set
            val_size (float): Proportion for validation set (from remaining data)
            random_state (int): Random seed
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: separate validation set from remaining data
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_feature_importance(self, X, y):
        """
        Calculate simple feature importance based on correlation with target.
        
        Args:
            X (pd.DataFrame): Features
            y (np.array): Target labels
            
        Returns:
            pd.DataFrame: Feature importance scores
        """
        # Calculate correlation with target
        correlations = []
        for col in X.columns:
            corr = np.corrcoef(X[col], y)[0, 1]
            correlations.append(abs(corr))
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': correlations
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_data_info(self, X, y=None):
        """
        Get information about the dataset.
        
        Args:
            X (pd.DataFrame): Features
            y (np.array, optional): Target labels
            
        Returns:
            dict: Dataset information
        """
        info = {
            'n_samples': len(X),
            'n_features': len(X.columns),
            'feature_names': X.columns.tolist(),
            'data_types': X.dtypes.to_dict(),
            'missing_values': X.isnull().sum().to_dict(),
            'feature_ranges': {}
        }
        
        # Add feature ranges for numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            info['feature_ranges'][col] = {
                'min': X[col].min(),
                'max': X[col].max(),
                'mean': X[col].mean(),
                'std': X[col].std()
            }
        
        # Add target information if provided
        if y is not None:
            info['target_distribution'] = {
                'normal': int(np.sum(y == 0)),
                'anomaly': int(np.sum(y == 1)),
                'anomaly_ratio': float(np.mean(y))
            }
        
        return info 