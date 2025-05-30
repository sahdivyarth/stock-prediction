import sys
import subprocess
import os

def check_python_environment():
    """Check Python environment and installed packages"""
    
    print("=== Python Environment Check ===")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    print()
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Running in a virtual environment")
        print(f"Virtual env path: {sys.prefix}")
    else:
        print("⚠️  Not running in a virtual environment")
        print("Recommendation: Create and activate a virtual environment")
    
    print()
    
    # Check installed packages
    print("=== Checking Installed Packages ===")
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                              capture_output=True, text=True)
        installed_packages = result.stdout
        
        # Check for required packages
        required_packages = ['Flask', 'flask-sqlalchemy', 'yfinance', 'pandas', 
                           'numpy', 'scikit-learn', 'tensorflow', 'statsmodels']
        
        for package in required_packages:
            if package.lower() in installed_packages.lower():
                print(f"✅ {package} is installed")
            else:
                print(f"❌ {package} is NOT installed")
        
        print("\n=== All Installed Packages ===")
        print(installed_packages)
        
    except Exception as e:
        print(f"Error checking packages: {e}")
    
    print("\n=== Installation Commands ===")
    print("To fix the issue, run these commands:")
    print("1. Create virtual environment:")
    print("   python -m venv venv")
    print("\n2. Activate virtual environment:")
    print("   Windows: venv\\Scripts\\activate")
    print("   Mac/Linux: source venv/bin/activate")
    print("\n3. Install requirements:")
    print("   pip install -r requirements.txt")
    print("\n4. Or install packages individually:")
    print("   pip install Flask Flask-SQLAlchemy yfinance pandas numpy scikit-learn tensorflow statsmodels plotly")

if __name__ == "__main__":
    check_python_environment()
