#!/usr/bin/env python3
"""
Simple script to run the Streamlit app with error checking.
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = ['streamlit', 'numpy', 'pandas', 'plotly', 'scikit-learn']
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {missing_packages}")
        print("Please install them using: pip install " + " ".join(missing_packages))
        return False
    
    return True

def run_streamlit_app():
    """Run the Streamlit app."""
    try:
        print("🚀 Starting Streamlit app...")
        print("📱 The app will open in your browser automatically")
        print("🔗 If it doesn't open, go to: http://localhost:8501")
        print("\n" + "="*50)
        
        # Run the simple test app first
        subprocess.run([sys.executable, "-m", "streamlit", "run", "test_simple_app.py"])
        
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {e}")

def main():
    """Main function."""
    print("🛡️ Network Anomaly Detection System")
    print("="*50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    print("\n🎯 Choose an app to run:")
    print("1. Test App (simple) - test_simple_app.py")
    print("2. Fixed App (simplified) - simple_app_fixed.py")
    print("3. Original App (full features) - simple_app.py")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    app_file = None
    if choice == "1":
        app_file = "test_simple_app.py"
    elif choice == "2":
        app_file = "simple_app_fixed.py"
    elif choice == "3":
        app_file = "simple_app.py"
    else:
        print("❌ Invalid choice. Running test app...")
        app_file = "test_simple_app.py"
    
    if not os.path.exists(app_file):
        print(f"❌ App file {app_file} not found!")
        return
    
    print(f"\n🎯 Running: {app_file}")
    run_streamlit_app()

if __name__ == "__main__":
    main() 