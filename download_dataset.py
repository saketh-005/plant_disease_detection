import os
import argparse
from huggingface_hub import hf_hub_download, snapshot_download

def download_dataset(
    repo_id: str = "saketh-005/plant-disease-dataset",
    save_dir: str = "dataset",
    token: str = None
):
    """
    Download dataset from Hugging Face Hub
    
    Args:
        repo_id: Repository ID on Hugging Face Hub (username/repo_name)
        save_dir: Directory to save the dataset
        token: Hugging Face authentication token (optional)
    """
    print(f"Downloading dataset from {repo_id}...")
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # Download the entire repository
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=save_dir,
            local_dir_use_symlinks=False,
            token=token
        )
        
        print(f"\nDataset downloaded successfully to: {os.path.abspath(save_dir)}")
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        print("\nMake sure you have access to the repository and provided the correct token if it's private.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download dataset from Hugging Face Hub')
    parser.add_argument('--repo_id', type=str, default='saketh-005/plant-disease-dataset',
                      help='Repository ID on Hugging Face Hub (username/repo_name)')
    parser.add_argument('--save_dir', type=str, default='dataset',
                      help='Directory to save the dataset')
    parser.add_argument('--token', type=str, default=None,
                      help='Hugging Face authentication token (required for private repos)')
    
    args = parser.parse_args()
    
    download_dataset(
        repo_id=args.repo_id,
        save_dir=args.save_dir,
        token=args.token
    )
