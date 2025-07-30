import streamlit as st
import numpy as np
import pandas as pd
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

def generate_sample_data(n_samples=5000, n_features=20, anomaly_ratio=0.15):
    """Generate sample network data."""
    np.random.seed(42)
    
    # Define realistic network traffic feature names
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
    
    # Use the first n_features from the list
    feature_names = network_features[:n_features]
    
    # Generate normal traffic
    n_normal = int(n_samples * (1 - anomaly_ratio))
    n_anomaly = n_samples - n_normal
    
    # Normal traffic patterns with realistic distributions
    normal_data = np.zeros((n_normal, n_features))
    
    for i, feature in enumerate(feature_names):
        if 'packet_size' in feature:
            # Packet size features (positive, log-normal distribution)
            normal_data[:, i] = np.random.lognormal(4, 0.5, n_normal)
        elif 'inter_arrival' in feature:
            # Inter-arrival time (positive, exponential distribution)
            normal_data[:, i] = np.random.exponential(1, n_normal)
        elif 'flow_' in feature:
            # Flow features (positive, log-normal distribution)
            normal_data[:, i] = np.random.lognormal(3, 0.8, n_normal)
        elif 'port' in feature:
            # Port features (integer-like, uniform distribution)
            normal_data[:, i] = np.random.randint(1, 65536, n_normal)
        elif 'protocol' in feature or 'service' in feature:
            # Protocol/service features (categorical-like)
            normal_data[:, i] = np.random.randint(0, 10, n_normal)
        elif 'count' in feature or 'num_' in feature:
            # Count features (integer-like, Poisson distribution)
            normal_data[:, i] = np.random.poisson(1, n_normal)
        elif 'rate' in feature:
            # Rate features (0-1, Beta distribution)
            normal_data[:, i] = np.random.beta(2, 5, n_normal)
        else:
            # Default normal distribution
            normal_data[:, i] = np.random.normal(0, 1, n_normal)
    
    # Anomalous traffic patterns (different distributions)
    anomaly_data = np.zeros((n_anomaly, n_features))
    
    for i, feature in enumerate(feature_names):
        if 'packet_size' in feature:
            # Anomalous packet sizes (much larger)
            anomaly_data[:, i] = np.random.lognormal(6, 1, n_anomaly)
        elif 'inter_arrival' in feature:
            # Anomalous inter-arrival times (much smaller - rapid packets)
            anomaly_data[:, i] = np.random.exponential(0.1, n_anomaly)
        elif 'flow_' in feature:
            # Anomalous flow features (much larger)
            anomaly_data[:, i] = np.random.lognormal(5, 1, n_anomaly)
        elif 'port' in feature:
            # Anomalous ports (well-known ports for attacks)
            anomaly_data[:, i] = np.random.randint(1, 1024, n_anomaly)
        elif 'count' in feature or 'num_' in feature:
            # Anomalous counts (much higher)
            anomaly_data[:, i] = np.random.poisson(10, n_anomaly)
        elif 'rate' in feature:
            # Anomalous rates (higher error rates)
            anomaly_data[:, i] = np.random.beta(5, 2, n_anomaly)
        else:
            # Default anomalous pattern (higher values)
            anomaly_data[:, i] = np.random.normal(3, 2, n_anomaly)
    
    # Combine data
    X = np.vstack([normal_data, anomaly_data])
    y = np.hstack([np.zeros(n_normal), np.ones(n_anomaly)])
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    return pd.DataFrame(X, columns=feature_names), y

def simple_autoencoder_predict(X, threshold=0.5):
    """Simple prediction function for demonstration."""
    # Simple reconstruction error calculation
    reconstructed = X * 0.9 + np.random.normal(0, 0.1, X.shape)
    mse = np.mean(np.square(X - reconstructed), axis=1)
    predictions = (mse > threshold).astype(int)
    return predictions, mse

def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🛡️ Network Anomaly Detection System")
    st.markdown("Simple autoencoder-based anomaly detection for network traffic")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Data generation parameters
        st.subheader("📊 Data Settings")
        n_samples = st.slider("Number of Samples", 1000, 10000, 5000)
        n_features = st.slider("Number of Features", 10, 30, 20)
        anomaly_ratio = st.slider("Anomaly Ratio", 0.05, 0.3, 0.15, 0.01)
        
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
                try:
                    # Generate data
                    X, y = generate_sample_data(n_samples, n_features, anomaly_ratio)
                    
                    # Simple preprocessing (just scaling)
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
                    
                    # Store in session state
                    st.session_state.data = {
                        'X': X,
                        'X_scaled': X_scaled,
                        'y': y,
                        'scaler': scaler,
                        'feature_names': X.columns.tolist()
                    }
                    st.session_state.model_trained = True
                    
                    st.success("✅ Model trained successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Display dataset information
        if st.session_state.data:
            data = st.session_state.data
            X, y = data['X'], data['y']
            
            # Dataset statistics
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("Total Samples", len(X))
            
            with col_b:
                st.metric("Features", len(X.columns))
            
            with col_c:
                st.metric("Anomaly Ratio", f"{np.mean(y):.3f}")
            
            # Data distribution
            import plotly.express as px
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
    
    with col2:
        st.header("🎯 Quick Info")
        
        if st.session_state.model_trained:
            st.success("✅ Model is ready for predictions!")
            st.info(f"Features: {len(st.session_state.data['feature_names'])}")
        else:
            st.info("Click 'Generate Data & Train Model' to get started")
    
    # Prediction interface
    if st.session_state.model_trained:
        st.header("🔮 Manual Prediction Interface")
        st.markdown("Enter feature values to predict if the network traffic is normal or anomalous.")
        
        # Get feature names
        feature_names = st.session_state.data['feature_names']
        
        # Create input form
        with st.form("prediction_form"):
            st.subheader("📝 Input Features")
            
            # Create input fields for each feature
            input_values = {}
            cols = st.columns(3)  # 3 columns for better layout
            
            for i, feature in enumerate(feature_names):
                col_idx = i % 3
                with cols[col_idx]:
                    # Get feature statistics
                    X = st.session_state.data['X']
                    feature_mean = X[feature].mean()
                    feature_std = X[feature].std()
                    
                    # Create better feature descriptions
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
                    
                    # Get description for this feature
                    description = feature_descriptions.get(feature, f"Network traffic feature: {feature}")
                    
                    # Format feature name for display
                    display_name = feature.replace('_', ' ').title()
                    
                    input_values[feature] = st.number_input(
                        f"{display_name}",
                        value=float(feature_mean),
                        min_value=float(feature_mean - 3*feature_std),
                        max_value=float(feature_mean + 3*feature_std),
                        step=float(feature_std/10),
                        help=f"{description}\nMean: {feature_mean:.3f}, Std: {feature_std:.3f}"
                    )
            
            # Submit button
            submitted = st.form_submit_button("🔍 Predict")
            
            if submitted:
                try:
                    # Create input dataframe
                    input_df = pd.DataFrame([input_values])
                    
                    # Scale input
                    scaler = st.session_state.data['scaler']
                    input_scaled = scaler.transform(input_df)
                    
                    # Make prediction
                    predictions, scores = simple_autoencoder_predict(input_scaled)
                    
                    # Display results
                    st.subheader("🎯 Prediction Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if predictions[0] == 1:
                            st.error("🚨 ANOMALY DETECTED")
                        else:
                            st.success("✅ NORMAL TRAFFIC")
                    
                    with col2:
                        st.metric("Prediction", "Anomaly" if predictions[0] == 1 else "Normal")
                    
                    with col3:
                        st.metric("Reconstruction Error", f"{scores[0]:.4f}")
                    
                    # Show input values
                    st.subheader("📋 Input Values")
                    input_df_display = input_df.copy()
                    input_df_display = input_df_display.round(3)
                    st.dataframe(input_df_display, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
        
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
                    
                    # Scale data
                    scaler = st.session_state.data['scaler']
                    batch_scaled = scaler.transform(batch_data)
                    
                    # Make predictions
                    predictions, scores = simple_autoencoder_predict(batch_scaled)
                    
                    # Display results
                    st.subheader("📊 Batch Prediction Results")
                    
                    results_df = pd.DataFrame({
                        'Sample': range(1, len(batch_data) + 1),
                        'Prediction': ['Anomaly' if p == 1 else 'Normal' for p in predictions],
                        'Reconstruction Error': scores,
                        'Confidence': ['High' if s > 0.5 else 'Low' for s in scores]
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
        <p>🛡️ Network Anomaly Detection System | Simple Autoencoder-Based Detection</p>
        <p>Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 