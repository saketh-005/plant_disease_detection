import os
import tarfile
from huggingface_hub import hf_hub_download

def download_dataset(save_dir="dataset"):
    """
    Download and extract the plant disease dataset
    """
    os.makedirs(save_dir, exist_ok=True)
    archive_path = os.path.join(save_dir, "plant-disease-dataset.tar.gz")
    
    print("Downloading dataset...")
    try:
        # Download the dataset
        hf_hub_download(
            repo_id="saketh-005/plant-disease-dataset",
            filename="plant-disease-dataset.tar.gz",
            local_dir=save_dir,
            local_dir_use_symlinks=False
        )
        
        # Extract the archive
        print("Extracting dataset...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=save_dir)
            
        # Remove the archive
        os.remove(archive_path)
        print("Dataset downloaded and extracted successfully!")
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    download_dataset()