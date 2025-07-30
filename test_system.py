#!/usr/bin/env python3
"""
Test script for the autoencoder anomaly detection system.
This script tests all components to ensure they work correctly.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def test_data_generation():
    """Test data generation functionality."""
    print("🧪 Testing data generation...")
    
    try:
        from data.sample_data import NetworkDataGenerator
        
        generator = NetworkDataGenerator(random_state=42)
        
        # Test basic data generation
        X, y = generator.generate_basic_network_data(1000, 20, 0.1)
        assert len(X) == 1000
        assert len(X.columns) == 20
        assert len(y) == 1000
        print("✅ Basic data generation: PASS")
        
        # Test complex data generation
        X, y = generator.generate_complex_network_data(1000, 25, 0.15)
        assert len(X) == 1000
        assert len(X.columns) == 25
        print("✅ Complex data generation: PASS")
        
        # Test realistic data generation
        X, y = generator.generate_realistic_network_features(1000, 0.1)
        assert len(X) == 1000
        print("✅ Realistic data generation: PASS")
        
        return True
        
    except Exception as e:
        print(f"❌ Data generation test failed: {e}")
        return False

def test_preprocessing():
    """Test data preprocessing functionality."""
    print("🧪 Testing data preprocessing...")
    
    try:
        from utils.data_preprocessing import NetworkDataPreprocessor
        from data.sample_data import NetworkDataGenerator
        
        # Generate test data
        generator = NetworkDataGenerator(random_state=42)
        X, y = generator.generate_basic_network_data(1000, 20, 0.1)
        
        # Test preprocessing
        preprocessor = NetworkDataPreprocessor(scaler_type='standard')
        X_processed, y_processed = preprocessor.fit_transform(X, y)
        
        assert X_processed.shape[0] == X.shape[0]
        assert len(X_processed.columns) >= len(X.columns)  # Feature engineering may add features
        print("✅ Data preprocessing: PASS")
        
        return True
        
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        return False

def test_autoencoder():
    """Test autoencoder model functionality."""
    print("🧪 Testing autoencoder model...")
    
    try:
        from models.autoencoder import Autoencoder
        
        # Create autoencoder
        autoencoder = Autoencoder(input_dim=20, encoding_dim=8, hidden_dims=[32, 16])
        
        # Build model
        model = autoencoder.build_model()
        assert model is not None
        print("✅ Autoencoder model creation: PASS")
        
        # Test training
        X_train = np.random.normal(0, 1, (100, 20))
        history = autoencoder.train(X_train, epochs=5, verbose=0)
        assert autoencoder.is_trained
        print("✅ Autoencoder training: PASS")
        
        # Test prediction
        X_test = np.random.normal(0, 1, (50, 20))
        predictions = autoencoder.predict(X_test)
        assert len(predictions) == 50
        print("✅ Autoencoder prediction: PASS")
        
        return True
        
    except Exception as e:
        print(f"❌ Autoencoder test failed: {e}")
        return False

def test_anomaly_detector():
    """Test anomaly detector wrapper."""
    print("🧪 Testing anomaly detector...")
    
    try:
        from models.anomaly_detector import AnomalyDetector
        from data.sample_data import NetworkDataGenerator
        
        # Generate test data
        generator = NetworkDataGenerator(random_state=42)
        X, y = generator.generate_basic_network_data(1000, 20, 0.1)
        
        # Create detector
        detector = AnomalyDetector(input_dim=20, encoding_dim=8, hidden_dims=[32, 16])
        
        # Train detector
        history = detector.fit(X, epochs=5, verbose=0)
        assert detector.is_fitted
        print("✅ Anomaly detector training: PASS")
        
        # Test prediction
        predictions = detector.predict(X)
        scores = detector.predict_proba(X)
        assert len(predictions) == len(X)
        assert len(scores) == len(X)
        print("✅ Anomaly detector prediction: PASS")
        
        return True
        
    except Exception as e:
        print(f"❌ Anomaly detector test failed: {e}")
        return False

def test_evaluation():
    """Test evaluation functionality."""
    print("🧪 Testing evaluation...")
    
    try:
        from utils.evaluation import AnomalyDetectionEvaluator
        from models.anomaly_detector import AnomalyDetector
        from data.sample_data import NetworkDataGenerator
        
        # Generate test data
        generator = NetworkDataGenerator(random_state=42)
        X, y = generator.generate_basic_network_data(1000, 20, 0.1)
        
        # Train detector
        detector = AnomalyDetector(input_dim=20, encoding_dim=8, hidden_dims=[32, 16])
        detector.fit(X, epochs=5, verbose=0)
        
        # Evaluate
        evaluator = AnomalyDetectionEvaluator()
        metrics = evaluator.evaluate_model(detector, X, y)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        print("✅ Evaluation: PASS")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation test failed: {e}")
        return False

def test_visualization():
    """Test visualization functionality."""
    print("🧪 Testing visualization...")
    
    try:
        from utils.visualization import AnomalyDetectionVisualizer
        from models.anomaly_detector import AnomalyDetector
        from data.sample_data import NetworkDataGenerator
        
        # Generate test data
        generator = NetworkDataGenerator(random_state=42)
        X, y = generator.generate_basic_network_data(1000, 20, 0.1)
        
        # Train detector
        detector = AnomalyDetector(input_dim=20, encoding_dim=8, hidden_dims=[32, 16])
        history = detector.fit(X, epochs=5, verbose=0)
        
        # Test visualization
        visualizer = AnomalyDetectionVisualizer()
        
        # Test training history plot
        fig = visualizer.plot_training_history(history)
        assert fig is not None
        print("✅ Training history visualization: PASS")
        
        # Test interactive plots
        fig_interactive = visualizer.plot_interactive_training_history(history)
        assert fig_interactive is not None
        print("✅ Interactive visualization: PASS")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False

def test_end_to_end():
    """Test complete end-to-end pipeline."""
    print("🧪 Testing end-to-end pipeline...")
    
    try:
        from models.anomaly_detector import AnomalyDetector
        from utils.data_preprocessing import NetworkDataPreprocessor
        from utils.evaluation import AnomalyDetectionEvaluator
        from data.sample_data import NetworkDataGenerator
        
        # Generate data
        generator = NetworkDataGenerator(random_state=42)
        X, y = generator.generate_complex_network_data(2000, 25, 0.15)
        
        # Preprocess data
        preprocessor = NetworkDataPreprocessor(scaler_type='standard')
        X_processed, y = preprocessor.fit_transform(X, y)
        
        # Train model
        detector = AnomalyDetector(
            input_dim=X_processed.shape[1],
            encoding_dim=16,
            hidden_dims=[64, 32]
        )
        
        # Train on normal data only
        normal_mask = y == 0
        X_normal = X_processed[normal_mask]
        
        history = detector.fit(X_normal, epochs=10, verbose=0)
        
        # Evaluate
        evaluator = AnomalyDetectionEvaluator()
        metrics = evaluator.evaluate_model(detector, X_processed, y)
        
        # Check that metrics are reasonable
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
        
        print("✅ End-to-end pipeline: PASS")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1-Score: {metrics['f1_score']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting System Tests")
    print("=" * 50)
    
    tests = [
        ("Data Generation", test_data_generation),
        ("Data Preprocessing", test_preprocessing),
        ("Autoencoder Model", test_autoencoder),
        ("Anomaly Detector", test_anomaly_detector),
        ("Evaluation", test_evaluation),
        ("Visualization", test_visualization),
        ("End-to-End Pipeline", test_end_to_end)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed!")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 