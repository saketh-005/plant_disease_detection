import os
from pathlib import Path
from huggingface_hub import HfApi, HfFolder

def upload_dataset_to_hf(
    dataset_path: str,
    repo_name: str = "plant-disease-dataset",
    repo_type: str = "dataset",
    private: bool = False
):
    """
    Upload a dataset to Hugging Face Hub
    
    Args:
        dataset_path: Path to the dataset directory
        repo_name: Name of the repository to create on Hugging Face Hub
        repo_type: Type of repository ('dataset' or 'model')
        private: Whether the repository should be private
    """
    # Initialize the Hugging Face API
    api = HfApi()
    
    # Get or create the repository
    repo_url = api.create_repo(
        repo_name,
        repo_type=repo_type,
        private=private,
        exist_ok=True
    )
    
    print(f"Uploading dataset to: {repo_url}")
    
    # Upload the dataset
    api.upload_folder(
        folder_path=dataset_path,
        repo_id=f"saketh-005/{repo_name}",
        repo_type=repo_type,
    )
    
    print(f"Successfully uploaded dataset to: {repo_url}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Upload dataset to Hugging Face Hub')
    parser.add_argument('--dataset_path', type=str, default='dataset',
                      help='Path to the dataset directory')
    parser.add_argument('--repo_name', type=str, default='plant-disease-dataset',
                      help='Name of the repository on Hugging Face Hub')
    parser.add_argument('--private', action='store_true',
                      help='Make the repository private')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset directory '{args.dataset_path}' not found!")
        exit(1)
    
    upload_dataset_to_hf(
        dataset_path=args.dataset_path,
        repo_name=args.repo_name,
        private=args.private
    )
