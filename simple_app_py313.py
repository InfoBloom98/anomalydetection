import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Network Anomaly Detection",
    page_icon="🛡️",
    layout="wide"
)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'data' not in st.session_state:
    st.session_state.data = {}
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

def generate_sample_data(n_samples=5000, anomaly_ratio=0.15, random_state=42):
    """Generate realistic network traffic data with meaningful feature names."""
    np.random.seed(random_state)
    
    # Define realistic network features
    network_features = [
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
    
    # Generate data with realistic distributions
    data = {}
    
    for feature in network_features:
        if 'packet_size' in feature:
            # Packet size features (bytes)
            if 'mean' in feature:
                data[feature] = np.random.normal(500, 200, n_samples)
            elif 'std' in feature:
                data[feature] = np.random.exponential(100, n_samples)
            elif 'min' in feature:
                data[feature] = np.random.uniform(64, 200, n_samples)
            elif 'max' in feature:
                data[feature] = np.random.uniform(800, 1500, n_samples)
        elif 'inter_arrival_time' in feature:
            # Inter-arrival time features (milliseconds)
            if 'mean' in feature:
                data[feature] = np.random.exponential(50, n_samples)
            elif 'std' in feature:
                data[feature] = np.random.exponential(20, n_samples)
            elif 'min' in feature:
                data[feature] = np.random.uniform(1, 10, n_samples)
        elif 'flow_' in feature:
            # Flow features
            if 'duration' in feature:
                data[feature] = np.random.exponential(100, n_samples)
            elif 'bytes' in feature:
                data[feature] = np.random.lognormal(8, 2, n_samples)
            elif 'packets' in feature:
                data[feature] = np.random.poisson(50, n_samples)
            elif 'rate' in feature:
                data[feature] = np.random.lognormal(6, 1.5, n_samples)
        elif 'port' in feature:
            # Port features
            data[feature] = np.random.randint(1, 65536, n_samples)
        elif 'protocol_type' in feature:
            data[feature] = np.random.randint(0, 3, n_samples)
        elif 'service_type' in feature:
            data[feature] = np.random.randint(0, 70, n_samples)
        elif 'count' in feature or 'num_' in feature:
            # Count features
            data[feature] = np.random.poisson(5, n_samples)
        elif 'rate' in feature:
            # Rate features (0-1)
            data[feature] = np.random.beta(2, 5, n_samples)
        elif 'logged_in' in feature or 'root_shell' in feature or 'su_attempted' in feature or 'is_host_login' in feature or 'is_guest_login' in feature:
            # Binary features
            data[feature] = np.random.binomial(1, 0.1, n_samples)
        else:
            # Default distribution
            data[feature] = np.random.normal(0, 1, n_samples)
    
    # Create DataFrame
    X = pd.DataFrame(data)
    
    # Generate anomalies by modifying some samples
    n_anomalies = int(n_samples * anomaly_ratio)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    
    # Create target variable
    y = np.zeros(n_samples)
    y[anomaly_indices] = 1
    
    # Modify anomalous samples to have unusual patterns
    for idx in anomaly_indices:
        # Increase packet sizes abnormally
        X.iloc[idx, X.columns.get_loc('packet_size_mean')] *= np.random.uniform(2, 5)
        X.iloc[idx, X.columns.get_loc('packet_size_max')] *= np.random.uniform(3, 8)
        
        # Increase flow duration abnormally
        X.iloc[idx, X.columns.get_loc('flow_duration')] *= np.random.uniform(5, 15)
        
        # Increase error rates
        X.iloc[idx, X.columns.get_loc('serror_rate')] = np.random.uniform(0.5, 1.0)
        X.iloc[idx, X.columns.get_loc('rerror_rate')] = np.random.uniform(0.3, 0.8)
        
        # Increase failed logins
        X.iloc[idx, X.columns.get_loc('num_failed_logins')] = np.random.poisson(20)
        
        # Set some binary flags
        X.iloc[idx, X.columns.get_loc('root_shell')] = 1
        X.iloc[idx, X.columns.get_loc('su_attempted')] = 1
    
    return X, y

def train_anomaly_detector(X_train, contamination=0.15):
    """Train an Isolation Forest model for anomaly detection."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Use Isolation Forest instead of autoencoder
    detector = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )
    
    detector.fit(X_train_scaled)
    
    return detector, scaler

def predict_anomalies(detector, scaler, X):
    """Predict anomalies using the trained model."""
    X_scaled = scaler.transform(X)
    predictions = detector.predict(X_scaled)
    
    # Isolation Forest returns -1 for anomalies, 1 for normal
    # Convert to 0/1 where 1 = anomaly, 0 = normal
    predictions = (predictions == -1).astype(int)
    
    # Get anomaly scores (lower = more anomalous)
    scores = detector.score_samples(X_scaled)
    
    return predictions, -scores  # Negate scores so higher = more anomalous

def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🛡️ Network Anomaly Detection System")
    st.markdown("Isolation Forest-based anomaly detection for network traffic")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Data generation parameters
        st.subheader("📊 Data Settings")
        n_samples = st.slider("Number of Samples", 1000, 10000, 5000)
        anomaly_ratio = st.slider("Anomaly Ratio", 0.05, 0.3, 0.15, 0.01)
        
        # Model parameters
        st.subheader("🤖 Model Settings")
        contamination = st.slider("Contamination", 0.05, 0.3, 0.15, 0.01)
        
        # Action buttons
        st.subheader("⚡ Actions")
        generate_and_train = st.button("🚀 Generate Data & Train Model", type="primary")
        
        if st.session_state.model_trained:
            st.success("✅ Model is trained and ready!")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Dataset & Training")
        
        if generate_and_train:
            with st.spinner("Generating data and training model..."):
                # Generate data
                X, y = generate_sample_data(n_samples=n_samples, anomaly_ratio=anomaly_ratio)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Train detector
                detector, scaler = train_anomaly_detector(X_train, contamination=contamination)
                
                # Make predictions on test set
                predictions, scores = predict_anomalies(detector, scaler, X_test)
                
                # Calculate metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                accuracy = accuracy_score(y_test, predictions)
                precision = precision_score(y_test, predictions, zero_division=0)
                recall = recall_score(y_test, predictions, zero_division=0)
                f1 = f1_score(y_test, predictions, zero_division=0)
                
                metrics = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
                
                # Store in session state
                st.session_state.detector = detector
                st.session_state.scaler = scaler
                st.session_state.data = {
                    'X': X,
                    'y': y,
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test,
                    'metrics': metrics
                }
                st.session_state.model_trained = True
                
                st.success("✅ Model trained successfully!")
        
        # Display dataset information
        if st.session_state.data:
            data = st.session_state.data
            X, y = data['X'], data['y']
            
            # Dataset statistics
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                st.metric("Total Samples", len(X))
            
            with col_b:
                st.metric("Features", len(X.columns))
            
            with col_c:
                st.metric("Anomaly Ratio", f"{np.mean(y):.3f}")
            
            with col_d:
                if 'metrics' in data:
                    st.metric("Model Accuracy", f"{data['metrics']['accuracy']:.3f}")
            
            # Data distribution
            fig = px.histogram(
                x=y,
                nbins=2,
                title="Data Distribution",
                labels={'x': 'Class', 'count': 'Count'},
                color_discrete_sequence=['blue']
            )
            fig.update_layout(
                xaxis=dict(
                    ticktext=['Normal', 'Anomaly'],
                    tickvals=[0, 1]
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Model performance if available
            if 'metrics' in data:
                st.subheader("📈 Model Performance")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Precision", f"{data['metrics']['precision']:.3f}")
                
                with col2:
                    st.metric("Recall", f"{data['metrics']['recall']:.3f}")
                
                with col3:
                    st.metric("F1-Score", f"{data['metrics']['f1_score']:.3f}")
                
                with col4:
                    st.metric("Accuracy", f"{data['metrics']['accuracy']:.3f}")
    
    with col2:
        st.header("🎯 Quick Info")
        
        if st.session_state.model_trained:
            st.success("✅ Model is ready for predictions!")
            
            # Model information
            st.subheader("🤖 Model Info")
            st.text("Algorithm: Isolation Forest")
            st.text("Features: 42 network features")
            st.text("Training samples: " + str(len(st.session_state.data['X_train'])))
        else:
            st.info("Click 'Generate Data & Train Model' to get started")
    
    # Prediction interface
    if st.session_state.model_trained:
        st.header("🔮 Manual Prediction Interface")
        st.markdown("Enter feature values to predict if the network traffic is normal or anomalous.")
        
        # Get feature names from the trained model
        data = st.session_state.data
        X = data['X']
        feature_names = X.columns.tolist()
        
        # Create input form
        with st.form("prediction_form"):
            st.subheader("📝 Input Features")
            
            # Create input fields for each feature
            input_values = {}
            cols = st.columns(3)  # 3 columns for better layout
            
            # Feature descriptions
            feature_descriptions = {
                'packet_size_mean': 'Average packet size in bytes',
                'packet_size_std': 'Standard deviation of packet sizes',
                'packet_size_min': 'Minimum packet size in bytes',
                'packet_size_max': 'Maximum packet size in bytes',
                'inter_arrival_time_mean': 'Average time between packets',
                'inter_arrival_time_std': 'Standard deviation of inter-arrival times',
                'inter_arrival_time_min': 'Minimum time between packets',
                'flow_duration': 'Duration of the network flow',
                'flow_bytes': 'Total bytes in the flow',
                'flow_packets': 'Total packets in the flow',
                'flow_rate': 'Flow rate (bytes per second)',
                'src_port': 'Source port number',
                'dst_port': 'Destination port number',
                'protocol_type': 'Protocol type identifier',
                'service_type': 'Service type identifier',
                'flag_count': 'Number of TCP flags',
                'urgent_count': 'Number of urgent packets',
                'hot_count': 'Number of hot indicators',
                'num_failed_logins': 'Number of failed login attempts',
                'logged_in': 'Login status (0/1)',
                'num_compromised': 'Number of compromised conditions',
                'root_shell': 'Root shell access (0/1)',
                'su_attempted': 'SU command attempted (0/1)',
                'num_root': 'Number of root accesses',
                'num_file_creations': 'Number of file creation operations',
                'num_shells': 'Number of shell prompts',
                'num_access_files': 'Number of operations on access files',
                'num_outbound_cmds': 'Number of outbound commands',
                'is_host_login': 'Login from same host (0/1)',
                'is_guest_login': 'Guest login (0/1)',
                'count': 'Number of connections to same host',
                'srv_count': 'Number of connections to same service',
                'serror_rate': 'Rate of SYN errors',
                'srv_serror_rate': 'Rate of SYN errors to same service',
                'rerror_rate': 'Rate of REJ errors',
                'srv_rerror_rate': 'Rate of REJ errors to same service',
                'same_srv_rate': 'Rate of connections to same service',
                'diff_srv_rate': 'Rate of connections to different services',
                'srv_diff_host_rate': 'Rate of connections to different hosts',
                'dst_host_count': 'Number of connections to destination host',
                'dst_host_srv_count': 'Number of connections to same service on destination',
                'dst_host_same_srv_rate': 'Rate of connections to same service on destination',
                'dst_host_diff_srv_rate': 'Rate of connections to different services on destination',
                'dst_host_same_src_port_rate': 'Rate of connections from same source port',
                'dst_host_srv_diff_host_rate': 'Rate of connections to same service from different hosts',
                'dst_host_serror_rate': 'Rate of SYN errors to destination host',
                'dst_host_srv_serror_rate': 'Rate of SYN errors to same service on destination',
                'dst_host_rerror_rate': 'Rate of REJ errors to destination host',
                'dst_host_srv_rerror_rate': 'Rate of REJ errors to same service on destination'
            }
            
            for i, feature in enumerate(feature_names):
                col_idx = i % 3
                with cols[col_idx]:
                    # Get feature statistics for better input ranges
                    feature_mean = X[feature].mean()
                    feature_std = X[feature].std()
                    feature_min = X[feature].min()
                    feature_max = X[feature].max()
                    
                    # Get description for this feature
                    description = feature_descriptions.get(feature, f"Network traffic feature: {feature}")
                    
                    # Format feature name for display
                    display_name = feature.replace('_', ' ').title()
                    
                    input_values[feature] = st.number_input(
                        f"{display_name}",
                        value=float(feature_mean),
                        min_value=float(feature_min - 2*feature_std),
                        max_value=float(feature_max + 2*feature_std),
                        step=float(feature_std/10),
                        help=f"{description}\nMean: {feature_mean:.3f}, Std: {feature_std:.3f}"
                    )
            
            # Submit button
            submitted = st.form_submit_button("🔍 Predict")
            
            if submitted:
                # Create input dataframe
                input_df = pd.DataFrame([input_values])
                
                # Make prediction
                prediction, score = predict_anomalies(st.session_state.detector, st.session_state.scaler, input_df)
                
                # Display results
                st.subheader("🎯 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction[0] == 1:
                        st.error("🚨 ANOMALY DETECTED")
                    else:
                        st.success("✅ NORMAL TRAFFIC")
                
                with col2:
                    st.metric("Prediction", "Anomaly" if prediction[0] == 1 else "Normal")
                
                with col3:
                    st.metric("Anomaly Score", f"{score[0]:.4f}")
                
                # Show input values
                st.subheader("📋 Input Values")
                input_df_display = input_df.copy()
                input_df_display = input_df_display.round(3)
                st.dataframe(input_df_display, use_container_width=True)
                
                # Show analysis
                st.subheader("🔍 Model Analysis")
                st.markdown(f"**Anomaly Score:** {score[0]:.4f}")
                
                if prediction[0] == 1:
                    st.warning("⚠️ High anomaly score indicates suspicious network activity")
                else:
                    st.info("ℹ️ Low anomaly score indicates normal network traffic")
        
        # Batch prediction option
        st.subheader("📊 Batch Prediction")
        st.markdown("Upload a CSV file with feature values for batch prediction.")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="CSV file should have the same feature columns as the training data"
        )
        
        if uploaded_file is not None:
            try:
                # Read uploaded file
                batch_data = pd.read_csv(uploaded_file)
                
                # Check if columns match
                expected_features = set(feature_names)
                uploaded_features = set(batch_data.columns)
                
                if expected_features.issubset(uploaded_features):
                    # Use only the expected features
                    batch_data = batch_data[feature_names]
                    
                    # Make predictions
                    predictions, scores = predict_anomalies(st.session_state.detector, st.session_state.scaler, batch_data)
                    
                    # Display results
                    st.subheader("📊 Batch Prediction Results")
                    
                    results_df = pd.DataFrame({
                        'Sample': range(1, len(batch_data) + 1),
                        'Prediction': ['Anomaly' if p == 1 else 'Normal' for p in predictions],
                        'Anomaly Score': scores,
                        'Confidence': ['High' if s > np.percentile(scores, 75) else 'Low' for s in scores]
                    })
                    
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Samples", len(batch_data))
                    
                    with col2:
                        st.metric("Anomalies Detected", np.sum(predictions))
                    
                    with col3:
                        st.metric("Anomaly Rate", f"{np.mean(predictions):.3f}")
                    
                else:
                    st.error(f"❌ Uploaded file must contain these features: {list(expected_features)}")
                    st.info(f"Uploaded file has: {list(uploaded_features)}")
                    
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🛡️ Network Anomaly Detection System | Isolation Forest-Based Detection</p>
        <p>Built with Streamlit, Scikit-learn, and Plotly</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 