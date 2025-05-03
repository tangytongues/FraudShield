import os
import shutil
import zipfile

def create_download_package():
    """
    Creates a ZIP file containing all the necessary files for running the application locally.
    """
    print("Creating download package...")
    
    # Define directories and files to include
    dirs_to_include = ['utils', 'pages', 'data']
    files_to_include = ['app.py', 'README.md']
    
    # Create a temporary directory for the package
    os.makedirs('download_package', exist_ok=True)
    
    # Copy directories
    for directory in dirs_to_include:
        if os.path.exists(directory):
            shutil.copytree(directory, f'download_package/{directory}', dirs_exist_ok=True)
        else:
            os.makedirs(f'download_package/{directory}', exist_ok=True)
    
    # Copy files
    for file in files_to_include:
        if os.path.exists(file):
            shutil.copy2(file, f'download_package/{file}')
    
    # Create a .streamlit directory with config.toml
    os.makedirs('download_package/.streamlit', exist_ok=True)
    with open('download_package/.streamlit/config.toml', 'w') as f:
        f.write("""[server]
headless = false
port = 8501
""")
    
    # Create requirements.txt
    with open('download_package/requirements.txt', 'w') as f:
        f.write("""streamlit==1.31.1
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.3
seaborn==0.13.0
plotly==5.17.0
scikit-learn==1.3.2
pillow==10.1.0
""")
    
    # Create ZIP file
    with zipfile.ZipFile('fraud_detection_system.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk('download_package'):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, 'download_package'))
    
    # Clean up temporary directory
    shutil.rmtree('download_package')
    
    print("Package created successfully: fraud_detection_system.zip")
    print("You can download this file and extract it on your local machine.")

if __name__ == "__main__":
    create_download_package()
