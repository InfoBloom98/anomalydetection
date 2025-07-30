import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class NetworkDataGenerator:
    """
    Generator for synthetic network traffic data.
    Creates realistic network traffic patterns with anomalies.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the data generator.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
    def generate_basic_network_data(self, n_samples=10000, n_features=20, anomaly_ratio=0.1):
        """
        Generate basic network traffic data with anomalies.
        
        Args:
            n_samples (int): Number of samples to generate
            n_features (int): Number of features
            anomaly_ratio (float): Ratio of anomalous samples
            
        Returns:
            tuple: (X, y) where X is features and y is labels
        """
        n_normal = int(n_samples * (1 - anomaly_ratio))
        n_anomaly = n_samples - n_normal
        
        # Generate normal traffic (low variance, centered around 0)
        normal_data = np.random.normal(0, 1, (n_normal, n_features))
        
        # Generate anomalous traffic (higher variance, different centers)
        anomaly_centers = np.random.uniform(2, 5, n_features)
        anomaly_data = np.random.normal(anomaly_centers, 2, (n_anomaly, n_features))
        
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
    
    def generate_complex_network_data(self, n_samples=10000, n_features=30, anomaly_ratio=0.15):
        """
        Generate more complex network traffic data with multiple anomaly types.
        
        Args:
            n_samples (int): Number of samples to generate
            n_features (int): Number of features
            anomaly_ratio (float): Ratio of anomalous samples
            
        Returns:
            tuple: (X, y) where X is features and y is labels
        """
        n_normal = int(n_samples * (1 - anomaly_ratio))
        n_anomaly = n_samples - n_normal
        
        # Generate normal traffic with some correlation structure
        normal_data = np.random.multivariate_normal(
            mean=np.zeros(n_features),
            cov=np.eye(n_features) * 0.8 + np.eye(n_features) * 0.2,
            size=n_normal
        )
        
        # Generate different types of anomalies
        n_anomaly_types = 3
        n_per_type = n_anomaly // n_anomaly_types
        
        anomaly_data = []
        
        # Type 1: High-volume attacks (high values in specific features)
        type1_data = np.random.normal(0, 1, (n_per_type, n_features))
        type1_data[:, :5] = np.random.normal(8, 2, (n_per_type, 5))  # High values in first 5 features
        anomaly_data.append(type1_data)
        
        # Type 2: DDoS-like patterns (high values across many features)
        type2_data = np.random.normal(6, 1.5, (n_per_type, n_features))
        anomaly_data.append(type2_data)
        
        # Type 3: Stealth attacks (subtle deviations)
        type3_data = np.random.normal(0, 1, (n_per_type, n_features))
        type3_data[:, 10:15] = np.random.normal(3, 0.5, (n_per_type, 5))  # Subtle changes
        anomaly_data.append(type3_data)
        
        # Combine all anomaly data
        all_anomaly_data = np.vstack(anomaly_data)
        
        # Combine normal and anomaly data
        X = np.vstack([normal_data, all_anomaly_data])
        y = np.hstack([np.zeros(n_normal), np.ones(n_anomaly)])
        
        # Shuffle the data
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]
        
        # Create feature names
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        return pd.DataFrame(X, columns=feature_names), y
    
    def generate_time_series_network_data(self, n_samples=10000, n_features=25, anomaly_ratio=0.1):
        """
        Generate network traffic data with temporal patterns.
        
        Args:
            n_samples (int): Number of samples to generate
            n_features (int): Number of features
            anomaly_ratio (float): Ratio of anomalous samples
            
        Returns:
            tuple: (X, y) where X is features and y is labels
        """
        n_normal = int(n_samples * (1 - anomaly_ratio))
        n_anomaly = n_samples - n_normal
        
        # Generate normal traffic with temporal patterns
        time_steps = np.linspace(0, 4*np.pi, n_normal)
        normal_data = np.zeros((n_normal, n_features))
        
        for i in range(n_features):
            # Add sinusoidal patterns with different frequencies
            frequency = 1 + (i % 3)
            normal_data[:, i] = np.sin(frequency * time_steps) + np.random.normal(0, 0.1, n_normal)
        
        # Generate anomalous traffic with different temporal patterns
        anomaly_data = np.zeros((n_anomaly, n_features))
        
        for i in range(n_features):
            # Anomalies have different temporal patterns
            if i < n_features // 3:
                # Sudden spikes
                anomaly_data[:, i] = np.random.normal(0, 1, n_anomaly) + np.random.choice([0, 5], n_anomaly, p=[0.8, 0.2])
            elif i < 2 * n_features // 3:
                # Gradual increases
                anomaly_data[:, i] = np.random.normal(3, 1, n_anomaly)
            else:
                # Oscillating patterns
                time_steps_anomaly = np.linspace(0, 2*np.pi, n_anomaly)
                anomaly_data[:, i] = 2 * np.sin(2 * time_steps_anomaly) + np.random.normal(0, 0.5, n_anomaly)
        
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
    
    def generate_realistic_network_features(self, n_samples=10000, anomaly_ratio=0.1):
        """
        Generate network traffic data with realistic feature names and patterns.
        
        Args:
            n_samples (int): Number of samples to generate
            anomaly_ratio (float): Ratio of anomalous samples
            
        Returns:
            tuple: (X, y) where X is features and y is labels
        """
        n_normal = int(n_samples * (1 - anomaly_ratio))
        n_anomaly = n_samples - n_normal
        
        # Define realistic network features
        feature_names = [
            'packet_size_mean', 'packet_size_std', 'packet_size_min', 'packet_size_max',
            'inter_arrival_time_mean', 'inter_arrival_time_std', 'inter_arrival_time_min',
            'flow_duration', 'flow_bytes', 'flow_packets', 'flow_rate',
            'src_port', 'dst_port', 'protocol_type', 'service_type',
            'flag_count', 'urgent_count', 'hot_count', 'num_failed_logins',
            'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
            'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
            'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
            'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
            'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
            'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
        ]
        
        n_features = len(feature_names)
        
        # Generate normal traffic patterns
        normal_data = np.zeros((n_normal, n_features))
        
        # Packet size features (positive values)
        normal_data[:, 0:4] = np.random.lognormal(4, 0.5, (n_normal, 4))
        
        # Inter-arrival time features (positive values)
        normal_data[:, 4:7] = np.random.exponential(1, (n_normal, 3))
        
        # Flow features (positive values)
        normal_data[:, 7:11] = np.random.lognormal(3, 0.8, (n_normal, 4))
        
        # Port features (integer-like)
        normal_data[:, 11:13] = np.random.randint(1, 65536, (n_normal, 2))
        
        # Protocol and service features (categorical-like)
        normal_data[:, 13:15] = np.random.randint(0, 10, (n_normal, 2))
        
        # Count features (integer-like)
        normal_data[:, 15:30] = np.random.poisson(1, (n_normal, 15))
        
        # Rate features (0-1)
        normal_data[:, 30:40] = np.random.beta(2, 5, (n_normal, 10))
        
        # Host features (integer-like)
        normal_data[:, 40:] = np.random.poisson(2, (n_normal, n_features - 40))
        
        # Generate anomalous traffic (different distributions)
        anomaly_data = np.zeros((n_anomaly, n_features))
        
        # Anomalies have different patterns
        for i in range(n_anomaly):
            # Randomly choose anomaly type
            anomaly_type = np.random.choice(3)
            
            if anomaly_type == 0:
                # High-volume attack
                anomaly_data[i] = normal_data[np.random.randint(0, n_normal)] * np.random.uniform(5, 10)
            elif anomaly_type == 1:
                # Port scan
                anomaly_data[i] = normal_data[np.random.randint(0, n_normal)]
                anomaly_data[i, 11:13] = np.random.randint(1, 1024, 2)  # Well-known ports
            else:
                # Protocol violation
                anomaly_data[i] = normal_data[np.random.randint(0, n_normal)]
                anomaly_data[i, 13:15] = np.random.randint(10, 20, 2)  # Unknown protocols
        
        # Combine data
        X = np.vstack([normal_data, anomaly_data])
        y = np.hstack([np.zeros(n_normal), np.ones(n_anomaly)])
        
        # Shuffle the data
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]
        
        return pd.DataFrame(X, columns=feature_names), y
    
    def add_noise_to_data(self, X, noise_level=0.1):
        """
        Add noise to the data to make it more realistic.
        
        Args:
            X (pd.DataFrame): Input data
            noise_level (float): Level of noise to add
            
        Returns:
            pd.DataFrame: Data with added noise
        """
        noise = np.random.normal(0, noise_level, X.shape)
        X_noisy = X + noise
        return X_noisy
    
    def create_data_summary(self, X, y):
        """
        Create a summary of the generated data.
        
        Args:
            X (pd.DataFrame): Feature data
            y (np.array): Target labels
            
        Returns:
            dict: Data summary
        """
        summary = {
            'n_samples': len(X),
            'n_features': len(X.columns),
            'feature_names': X.columns.tolist(),
            'target_distribution': {
                'normal': int(np.sum(y == 0)),
                'anomaly': int(np.sum(y == 1)),
                'anomaly_ratio': float(np.mean(y))
            },
            'feature_statistics': {
                'mean': X.mean().to_dict(),
                'std': X.std().to_dict(),
                'min': X.min().to_dict(),
                'max': X.max().to_dict()
            }
        }
        
        return summary 