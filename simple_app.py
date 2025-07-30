import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from models.anomaly_detector import AnomalyDetector
from utils.data_preprocessing import NetworkDataPreprocessor
from utils.evaluation import AnomalyDetectionEvaluator
from data.sample_data import NetworkDataGenerator

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

def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🛡️ Network Anomaly Detection System")
    st.markdown("Autoencoder-based anomaly detection for network traffic")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Data generation parameters
        st.subheader("📊 Data Settings")
        n_samples = st.slider("Number of Samples", 1000, 10000, 5000)
        n_features = st.slider("Number of Features", 10, 30, 20)
        anomaly_ratio = st.slider("Anomaly Ratio", 0.05, 0.3, 0.15, 0.01)
        
        # Model parameters
        st.subheader("🤖 Model Settings")
        encoding_dim = st.slider("Encoding Dimension", 8, 32, 16)
        epochs = st.slider("Training Epochs", 10, 100, 50)
        
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
                # Generate data with realistic network features
                data_generator = NetworkDataGenerator(random_state=42)
                X, y = data_generator.generate_realistic_network_features(
                    n_samples=n_samples,
                    anomaly_ratio=anomaly_ratio
                )
                
                # Preprocess data
                preprocessor = NetworkDataPreprocessor(scaler_type='standard')
                X_processed, y = preprocessor.fit_transform(X, y)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Train on normal data only
                normal_mask = y_train == 0
                X_train_normal = X_train[normal_mask]
                
                # Train detector
                detector = AnomalyDetector(
                    input_dim=X_train.shape[1],
                    encoding_dim=encoding_dim,
                    hidden_dims=[64, 32]
                )
                
                history = detector.fit(
                    X_train_normal,
                    epochs=epochs,
                    batch_size=32,
                    verbose=0
                )
                
                # Evaluate model
                evaluator = AnomalyDetectionEvaluator()
                metrics = evaluator.evaluate_model(detector, X_test, y_test)
                
                # Store in session state
                st.session_state.detector = detector
                st.session_state.data = {
                    'X': X,
                    'X_processed': X_processed,
                    'y': y,
                    'preprocessor': preprocessor,
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
                    st.metric("ROC AUC", f"{data['metrics']['roc_auc']:.3f}")
    
    with col2:
        st.header("🎯 Quick Info")
        
        if st.session_state.model_trained:
            st.success("✅ Model is ready for predictions!")
            
            # Model information
            model_info = st.session_state.detector.get_model_info()
            st.subheader("🤖 Model Info")
            for key, value in model_info.items():
                st.text(f"{key}: {value}")
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
            
            for i, feature in enumerate(feature_names):
                col_idx = i % 3
                with cols[col_idx]:
                    # Get feature statistics for better input ranges
                    feature_mean = X[feature].mean()
                    feature_std = X[feature].std()
                    feature_min = X[feature].min()
                    feature_max = X[feature].max()
                    
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
                
                # Preprocess input using the same preprocessor
                preprocessor = data['preprocessor']
                input_processed = preprocessor.transform(input_df)
                
                # Make prediction
                prediction = st.session_state.detector.predict(input_processed)
                score = st.session_state.detector.predict_proba(input_processed)
                
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
                    st.metric("Confidence Score", f"{score[0]:.4f}")
                
                # Show input values
                st.subheader("📋 Input Values")
                input_df_display = input_df.copy()
                input_df_display = input_df_display.round(3)
                st.dataframe(input_df_display, use_container_width=True)
                
                # Show reconstruction error
                st.subheader("🔍 Model Analysis")
                st.markdown(f"**Reconstruction Error:** {score[0]:.4f}")
                st.markdown(f"**Threshold:** {st.session_state.detector.get_threshold():.4f}")
                
                if score[0] > st.session_state.detector.get_threshold():
                    st.warning("⚠️ High reconstruction error indicates anomalous pattern")
                else:
                    st.info("ℹ️ Low reconstruction error indicates normal pattern")
        
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
                    
                    # Preprocess
                    preprocessor = data['preprocessor']
                    batch_processed = preprocessor.transform(batch_data)
                    
                    # Make predictions
                    predictions = st.session_state.detector.predict(batch_processed)
                    scores = st.session_state.detector.predict_proba(batch_processed)
                    
                    # Display results
                    st.subheader("📊 Batch Prediction Results")
                    
                    results_df = pd.DataFrame({
                        'Sample': range(1, len(batch_data) + 1),
                        'Prediction': ['Anomaly' if p == 1 else 'Normal' for p in predictions],
                        'Reconstruction Error': scores,
                        'Confidence': ['High' if s > st.session_state.detector.get_threshold() else 'Low' for s in scores]
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
        <p>🛡️ Network Anomaly Detection System | Autoencoder-Based Detection</p>
        <p>Built with Streamlit, TensorFlow, and Plotly</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 