#!/usr/bin/env python3
"""
Installation script for the Network Anomaly Detection System.
Handles Python version compatibility and dependency installation.
"""

import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        return False
    
    if version.major == 3 and version.minor >= 13:
        print("⚠️  Python 3.13 detected - using compatible package versions")
        return "3.13"
    
    print("✅ Python version is compatible")
    return True

def install_package(package):
    """Install a single package."""
    try:
        print(f"📦 Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def install_dependencies():
    """Install all required dependencies."""
    print("🚀 Installing Network Anomaly Detection Dependencies")
    print("=" * 60)
    
    # Check Python version
    version_check = check_python_version()
    if not version_check:
        return False
    
    # Define packages based on Python version
    if version_check == "3.13":
        # Python 3.13 compatible packages
        packages = [
            "numpy>=1.26.0",
            "pandas>=2.1.0", 
            "scikit-learn>=1.3.0",
            "streamlit>=1.28.0",
            "plotly>=5.17.0"
        ]
    else:
        # Standard packages for older Python versions
        packages = [
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "scikit-learn>=1.3.0", 
            "streamlit>=1.25.0",
            "plotly>=5.15.0"
        ]
    
    # Install packages
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n📊 Installation Summary: {success_count}/{len(packages)} packages installed successfully")
    
    if success_count == len(packages):
        print("🎉 All dependencies installed successfully!")
        return True
    else:
        print("⚠️ Some packages failed to install. You may need to install them manually.")
        return False

def test_imports():
    """Test if all required packages can be imported."""
    print("\n🧪 Testing package imports...")
    
    required_packages = ['numpy', 'pandas', 'sklearn', 'streamlit', 'plotly']
    failed_imports = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import {package}: {e}")
            failed_imports.append(package)
    
    if not failed_imports:
        print("🎉 All packages imported successfully!")
        return True
    else:
        print(f"⚠️ Failed to import: {failed_imports}")
        return False

def main():
    """Main installation function."""
    print("🛡️ Network Anomaly Detection System - Dependency Installer")
    print("=" * 60)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Installation failed!")
        return False
    
    # Test imports
    if not test_imports():
        print("❌ Import test failed!")
        return False
    
    print("\n🎯 Installation completed successfully!")
    print("You can now run the app with:")
    print("  streamlit run simple_app_fixed.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 