import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from models.anomaly_detector import AnomalyDetector
from utils.data_preprocessing import NetworkDataPreprocessor
from utils.evaluation import AnomalyDetectionEvaluator
from utils.visualization import AnomalyDetectionVisualizer
from data.sample_data import NetworkDataGenerator

# Page configuration
st.set_page_config(
    page_title="Network Threat Detection - Autoencoder Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'evaluator' not in st.session_state:
    st.session_state.evaluator = None
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = None
if 'data' not in st.session_state:
    st.session_state.data = {}
if 'training_history' not in st.session_state:
    st.session_state.training_history = None

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🛡️ Network Threat Detection</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #666;">Autoencoder-Based Anomaly Detection System</h2>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # Data generation options
        st.subheader("📊 Data Configuration")
        data_type = st.selectbox(
            "Data Type",
            ["Basic Network Data", "Complex Network Data", "Time Series Data", "Realistic Network Features"]
        )
        
        n_samples = st.slider("Number of Samples", 1000, 20000, 10000)
        n_features = st.slider("Number of Features", 10, 50, 25)
        anomaly_ratio = st.slider("Anomaly Ratio", 0.05, 0.3, 0.15, 0.01)
        
        # Model configuration
        st.subheader("🤖 Model Configuration")
        encoding_dim = st.slider("Encoding Dimension", 8, 64, 16)
        hidden_dims = st.multiselect(
            "Hidden Layer Dimensions",
            [32, 64, 128, 256],
            default=[64, 32]
        )
        
        # Training configuration
        st.subheader("🏋️ Training Configuration")
        epochs = st.slider("Epochs", 10, 200, 50)
        batch_size = st.slider("Batch Size", 16, 128, 32)
        
        # Action buttons
        st.subheader("⚡ Actions")
        generate_data = st.button("📊 Generate Data", type="primary")
        train_model = st.button("🤖 Train Model", type="primary")
        evaluate_model = st.button("📈 Evaluate Model", type="primary")
        load_model = st.button("📁 Load Model")
        save_model = st.button("💾 Save Model")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Data Overview")
        
        if generate_data:
            with st.spinner("Generating network traffic data..."):
                # Generate data
                data_generator = NetworkDataGenerator(random_state=42)
                
                if data_type == "Basic Network Data":
                    X, y = data_generator.generate_basic_network_data(n_samples, n_features, anomaly_ratio)
                elif data_type == "Complex Network Data":
                    X, y = data_generator.generate_complex_network_data(n_samples, n_features, anomaly_ratio)
                elif data_type == "Time Series Data":
                    X, y = data_generator.generate_time_series_network_data(n_samples, n_features, anomaly_ratio)
                else:  # Realistic Network Features
                    X, y = data_generator.generate_realistic_network_features(n_samples, anomaly_ratio)
                
                # Preprocess data
                preprocessor = NetworkDataPreprocessor(scaler_type='standard')
                X_processed, y = preprocessor.fit_transform(X, y)
                
                # Store in session state
                st.session_state.data = {
                    'X': X,
                    'X_processed': X_processed,
                    'y': y,
                    'preprocessor': preprocessor
                }
                st.session_state.preprocessor = preprocessor
                
                st.success("✅ Data generated and preprocessed successfully!")
        
        # Display data information
        if st.session_state.data:
            data = st.session_state.data
            X, y = data['X'], data['y']
            
            # Data statistics
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("Total Samples", len(X))
            
            with col_b:
                st.metric("Features", len(X.columns))
            
            with col_c:
                st.metric("Anomaly Ratio", f"{np.mean(y):.3f}")
            
            # Data distribution plot
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
            
            # Feature statistics
            st.subheader("📈 Feature Statistics")
            feature_stats = pd.DataFrame({
                'Mean': X.mean(),
                'Std': X.std(),
                'Min': X.min(),
                'Max': X.max()
            })
            st.dataframe(feature_stats, use_container_width=True)
    
    with col2:
        st.header("🎯 Quick Actions")
        
        if st.session_state.data:
            st.success("✅ Data loaded")
            
            if train_model:
                with st.spinner("Training autoencoder model..."):
                    # Split data
                    from sklearn.model_selection import train_test_split
                    X_processed = st.session_state.data['X_processed']
                    y = st.session_state.data['y']
                    
                    X_train, X_temp, y_train, y_temp = train_test_split(
                        X_processed, y, test_size=0.3, random_state=42, stratify=y
                    )
                    
                    # Train on normal data only
                    normal_mask = y_train == 0
                    X_train_normal = X_train[normal_mask]
                    
                    # Initialize and train detector
                    detector = AnomalyDetector(
                        input_dim=X_train.shape[1],
                        encoding_dim=encoding_dim,
                        hidden_dims=hidden_dims
                    )
                    
                    history = detector.fit(
                        X_train_normal,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=0
                    )
                    
                    # Store in session state
                    st.session_state.detector = detector
                    st.session_state.training_history = history
                    
                    st.success("✅ Model trained successfully!")
        
        if st.session_state.detector:
            st.success("✅ Model ready")
            
            # Model information
            model_info = st.session_state.detector.get_model_info()
            st.subheader("🤖 Model Info")
            for key, value in model_info.items():
                st.text(f"{key}: {value}")
    
    # Model training and evaluation section
    if st.session_state.detector and st.session_state.training_history:
        st.header("📈 Model Performance")
        
        # Evaluate model
        if evaluate_model:
            with st.spinner("Evaluating model performance..."):
                # Split data for evaluation
                X_processed = st.session_state.data['X_processed']
                y = st.session_state.data['y']
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Evaluate
                evaluator = AnomalyDetectionEvaluator()
                metrics = evaluator.evaluate_model(st.session_state.detector, X_test, y_test)
                
                st.session_state.evaluator = evaluator
                
                # Display metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                
                with col2:
                    st.metric("Precision", f"{metrics['precision']:.4f}")
                
                with col3:
                    st.metric("Recall", f"{metrics['recall']:.4f}")
                
                with col4:
                    st.metric("F1-Score", f"{metrics['f1_score']:.4f}")
                
                with col5:
                    st.metric("ROC AUC", f"{metrics['roc_auc']:.4f}")
                
                # Create visualizations
                visualizer = AnomalyDetectionVisualizer()
                st.session_state.visualizer = visualizer
                
                # Training history
                fig_history = visualizer.plot_interactive_training_history(st.session_state.training_history)
                st.plotly_chart(fig_history, use_container_width=True)
                
                # Confusion matrix
                fig_cm = visualizer.plot_confusion_matrix_interactive(metrics['confusion_matrix'])
                st.plotly_chart(fig_cm, use_container_width=True)
                
                # ROC curve
                y_true = evaluator.predictions['y_true']
                y_scores = evaluator.predictions['y_scores']
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                fig_roc = visualizer.plot_roc_curve_interactive(fpr, tpr, metrics['roc_auc'])
                st.plotly_chart(fig_roc, use_container_width=True)
    
    # Real-time prediction section
    st.header("🔮 Real-Time Prediction")
    
    if st.session_state.detector:
        st.subheader("Test the Model")
        
        # Generate sample data for prediction
        if st.button("Generate Test Sample"):
            data_generator = NetworkDataGenerator(random_state=123)
            X_test_sample, y_test_sample = data_generator.generate_basic_network_data(
                n_samples=100, n_features=20, anomaly_ratio=0.2
            )
            
            # Use the detector's own scaler for consistency
            # First, we need to engineer features to match the training data format
            preprocessor = NetworkDataPreprocessor(scaler_type='standard')
            X_test_engineered = preprocessor.engineer_features(X_test_sample)
            
            # Then use the detector's scaler
            X_test_processed = st.session_state.detector.scaler.transform(X_test_engineered)
            X_test_processed = pd.DataFrame(X_test_processed, columns=X_test_engineered.columns)
            
            # Make predictions
            predictions = st.session_state.detector.predict(X_test_processed)
            scores = st.session_state.detector.predict_proba(X_test_processed)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Samples", len(X_test_sample))
            
            with col2:
                st.metric("Predicted Anomalies", np.sum(predictions))
            
            with col3:
                st.metric("True Anomalies", np.sum(y_test_sample))
            
            # Show sample predictions
            st.subheader("Sample Predictions")
            results_df = pd.DataFrame({
                'Sample': range(1, len(X_test_sample) + 1),
                'True Label': ['Anomaly' if y == 1 else 'Normal' for y in y_test_sample],
                'Predicted Label': ['Anomaly' if p == 1 else 'Normal' for p in predictions],
                'Reconstruction Error': scores,
                'Correct': predictions == y_test_sample
            })
            
            st.dataframe(results_df.head(10), use_container_width=True)
            
            # Accuracy
            accuracy = np.mean(predictions == y_test_sample)
            st.metric("Prediction Accuracy", f"{accuracy:.4f}")
    
    # Model management section
    st.header("💾 Model Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.detector:
            if save_model:
                with st.spinner("Saving model..."):
                    st.session_state.detector.save_model('models/anomaly_detector')
                    st.success("✅ Model saved successfully!")
    
    with col2:
        if load_model:
            with st.spinner("Loading model..."):
                try:
                    detector = AnomalyDetector()
                    detector.load_model('models/anomaly_detector')
                    st.session_state.detector = detector
                    st.success("✅ Model loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Error loading model: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🛡️ Network Threat Detection System | Autoencoder-Based Anomaly Detection</p>
        <p>Built with Streamlit, TensorFlow, and Plotly</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 