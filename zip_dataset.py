import os
import zipfile
import shutil
from pathlib import Path

def zip_dataset(dataset_dir="plant_disease_dataset", output_zip="plant_disease_dataset.zip"):
    """
    Zip the plant disease dataset for Kaggle upload.
    
    Args:
        dataset_dir (str): Path to the dataset directory
        output_zip (str): Name of the output zip file
    """
    # Convert to absolute paths
    dataset_path = Path(dataset_dir).absolute()
    zip_path = Path(output_zip).absolute()
    
    print(f"Preparing to zip dataset from: {dataset_path}")
    print(f"Output zip file will be: {zip_path}")
    
    # Verify the dataset directory exists
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
    
    # Check if the required subdirectories exist
    required_dirs = {'train', 'valid', 'test'}
    subdirs = {d.name for d in dataset_path.iterdir() if d.is_dir()}
    missing_dirs = required_dirs - subdirs
    
    if missing_dirs:
        print(f"Warning: The following required subdirectories are missing: {', '.join(missing_dirs)}")
    
    # Remove existing zip file if it exists
    if zip_path.exists():
        print(f"Removing existing zip file: {zip_path}")
        zip_path.unlink()
    
    # Create a temporary directory for the zip file
    temp_dir = Path("temp_zip")
    if temp_dir.exists():
        print(f"Cleaning up existing temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)
    
    # Create a copy of the dataset in the temp directory and clean filenames
    print("Creating temporary copy of the dataset and cleaning filenames...")
    temp_dataset = temp_dir / dataset_path.name
    
    # First copy the entire directory structure
    shutil.copytree(dataset_path, temp_dataset)
    
    def clean_filename(filename):
        """Clean up filename by removing spaces before extension and handling 'copy' in filenames"""
        # Split into name and extension
        name, ext = os.path.splitext(filename)
        
        # Remove spaces before extension and ' copy' before the extension
        name = name.rstrip()
        if ' copy' in name.lower():
            name = name.lower().replace(' copy', '').strip()
        
        # Clean up any remaining spaces before dots
        name = name.replace(' .', '.')
        
        # Remove any trailing dots or spaces
        name = name.rstrip('. ')
        
        # Reconstruct filename
        return f"{name}{ext}".strip()
    
    # Function to clean filenames in directory
    def clean_filenames(root_dir):
        for root, dirs, files in os.walk(root_dir, topdown=False):
            # Clean directory names
            for dir_name in dirs[:]:  # Make a copy of the list since we're modifying it
                old_path = os.path.join(root, dir_name)
                new_name = clean_filename(dir_name)
                if new_name != dir_name:
                    new_path = os.path.join(root, new_name)
                    try:
                        os.rename(old_path, new_path)
                        print(f"Renamed directory: {os.path.relpath(old_path, root_dir)} -> {os.path.relpath(new_path, root_dir)}")
                    except OSError as e:
                        print(f"Error renaming directory {old_path}: {e}")
            
            # Clean file names
            for file_name in files:
                old_path = os.path.join(root, file_name)
                new_name = clean_filename(file_name)
                if new_name != file_name:
                    new_path = os.path.join(root, new_name)
                    try:
                        os.rename(old_path, new_path)
                        print(f"Renamed file: {os.path.relpath(old_path, root_dir)} -> {os.path.relpath(new_path, root_dir)}")
                    except OSError as e:
                        print(f"Error renaming file {old_path}: {e}")
    
    # Clean filenames in the copied directory
    print("Cleaning up filenames...")
    clean_filenames(temp_dataset)
    
    # Create zip file
    print(f"Creating zip archive: {zip_path}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_dataset):
            for file in files:
                file_path = Path(root) / file
                # Calculate relative path for the zip file
                rel_path = file_path.relative_to(temp_dir)
                print(f"Adding: {rel_path}")
                zipf.write(file_path, rel_path)
    
    # Clean up
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)
    
    # Verify the zip file was created
    if zip_path.exists():
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"\nSuccessfully created zip file: {zip_path}")
        print(f"Size: {size_mb:.2f} MB")
        print("\nYou can now upload this zip file to Kaggle as a new dataset.")
    else:
        print("Error: Failed to create zip file")

if __name__ == "__main__":
    zip_dataset()
