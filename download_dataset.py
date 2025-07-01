import os
import tarfile
import urllib.request
import shutil

def download_file(url, filename):
    """Download a file from URL with progress"""
    with urllib.request.urlopen(url) as response, open(filename, 'wb') as out_file:
        total_size = int(response.getheader('content-length', 0))
        block_size = 1024 * 8  # 8KB blocks
        downloaded = 0
        
        print(f"Downloading {os.path.basename(filename)}...")
        while True:
            buffer = response.read(block_size)
            if not buffer:
                break
            out_file.write(buffer)
            downloaded += len(buffer)
            if total_size > 0:
                percent = min(100, int(downloaded * 100 / total_size))
                print(f"\rProgress: {downloaded//(1024*1024)}MB/{total_size//(1024*1024)}MB ({percent}%)", end='')
        print()  # New line after progress

def download_dataset(save_dir="dataset"):
    """
    Download and extract the plant disease dataset from the provided URL
    """
    os.makedirs(save_dir, exist_ok=True)
    dataset_url = "https://huggingface.co/datasets/saketh-005/plant-disease-dataset/resolve/main/plant-disease-dataset.tar.gz"
    archive_path = os.path.join(save_dir, "plant-disease-dataset.tar.gz")
    
    try:
        # Download the dataset
        download_file(dataset_url, archive_path)
        
        # Extract the archive
        print("\nExtracting dataset...")
        with tarfile.open(archive_path, "r:gz") as tar:
            # Get the root directory name
            root_dir = tar.getnames()[0].split('/')[0] if tar.getnames() else ''
            # Extract all files
            tar.extractall(path=save_dir)
            
        # Move files from the subdirectory if needed
        extracted_dir = os.path.join(save_dir, root_dir) if root_dir else save_dir
        if os.path.exists(extracted_dir) and extracted_dir != save_dir:
            for item in os.listdir(extracted_dir):
                shutil.move(os.path.join(extracted_dir, item), save_dir)
            os.rmdir(extracted_dir)
            
        # Remove the archive
        os.remove(archive_path)
        print("\nDataset downloaded and extracted successfully!")
        print(f"Files saved to: {os.path.abspath(save_dir)}")
        return True
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        # Clean up partially downloaded file if exists
        if os.path.exists(archive_path):
            os.remove(archive_path)
        return False

if __name__ == "__main__":
    download_dataset()