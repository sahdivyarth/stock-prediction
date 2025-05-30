import zipfile
import os
from pathlib import Path

def create_project_zip():
    """Create a ZIP file containing the entire project"""
    
    # Define the project structure
    files_to_include = [
        'app.py',
        'models.py', 
        'prediction.py',
        'requirements.txt',
        'README.md',
        'static/css/style.css',
        'static/js/main.js',
        'templates/base.html',
        'templates/index.html',
        'templates/auth.html',
        'templates/dashboard.html',
        'templates/history.html'
    ]
    
    # Create directories if they don't exist
    directories = [
        'static/css',
        'static/js', 
        'templates',
        'instance'
    ]
    
    zip_filename = 'stock-prediction-app.zip'
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add all project files
        for file_path in files_to_include:
            if os.path.exists(file_path):
                zipf.write(file_path, file_path)
                print(f"Added: {file_path}")
            else:
                print(f"Warning: {file_path} not found")
        
        # Add additional configuration files
        additional_files = [
            ('.env.example', """SECRET_KEY=your-secret-key-change-in-production
DATABASE_URL=sqlite:///stock_app.db
FLASK_ENV=development"""),
            
            ('Procfile', 'web: gunicorn app:app'),
            
            ('runtime.txt', 'python-3.9.18'),
            
            ('.gitignore', """__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/
instance/
*.db
.DS_Store"""),
            
            ('Dockerfile', """FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]"""),
            
            ('docker-compose.yml', """version: '3.8'
services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - SECRET_KEY=your-secret-key-here
    volumes:
      - ./instance:/app/instance""")
        ]
        
        for filename, content in additional_files:
            zipf.writestr(filename, content)
            print(f"Added: {filename}")
    
    print(f"\n✅ Project ZIP file created: {zip_filename}")
    print(f"📦 File size: {os.path.getsize(zip_filename) / 1024 / 1024:.2f} MB")
    
    return zip_filename

if __name__ == "__main__":
    create_project_zip()
