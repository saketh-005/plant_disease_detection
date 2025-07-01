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

def download_dataset(save_dir="plant_disease_dataset"):
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
            members = []
            for member in tar.getmembers():
                # Skip macOS resource fork files (._*) and other special files
                if not any(part.startswith('._') for part in member.name.split('/')):
                    # Ensure the target path is within the destination directory for security
                    member_path = os.path.join(save_dir, member.name)
                    if not os.path.abspath(member_path).startswith(os.path.abspath(save_dir)):
                        continue
                    members.append(member)
            
            # Extract only the filtered members
            if members:
                tar.extractall(path=save_dir, members=members, filter='data')
                
                # Find the root directory of the extracted files
                root_dirs = set()
                for member in members:
                    first_part = member.name.split('/')[0]
                    if first_part and first_part != '.':
                        root_dirs.add(first_part)
                
                # If there's a single root directory, move its contents up
                if len(root_dirs) == 1:
                    root_dir = root_dirs.pop()
                    extracted_dir = os.path.join(save_dir, root_dir)
                    if os.path.exists(extracted_dir) and os.path.isdir(extracted_dir):
                        for item in os.listdir(extracted_dir):
                            src = os.path.join(extracted_dir, item)
                            dst = os.path.join(save_dir, item)
                            if os.path.exists(dst):
                                if os.path.isdir(dst):
                                    shutil.rmtree(dst)
                                else:
                                    os.remove(dst)
                            shutil.move(src, save_dir)
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