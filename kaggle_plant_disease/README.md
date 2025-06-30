# Plant Disease Classification on Kaggle

This directory contains everything you need to train a plant disease classification model on Kaggle with GPU acceleration.

## Files

- `kaggle_train.py`: Main training script optimized for Kaggle
- `requirements.txt`: Required Python packages
- `download_model.py`: Script to download the trained model and class indices

## How to Use

1. **Prepare Your Dataset**
   - Compress your dataset folder (containing train/valid subdirectories) into a ZIP file
   - The expected structure is:
     ```
     your_dataset.zip
     └── New Plant Diseases Dataset(Augmented)
         └── New Plant Diseases Dataset(Augmented)
             ├── train
             │   ├── class1
             │   ├── class2
             │   └── ...
             └── valid
                 ├── class1
                 ├── class2
                 └── ...
     ```

2. **Upload to Kaggle**
   - Go to [Kaggle.com](https://www.kaggle.com/)
   - Create a new notebook
   - Click "Add Data" → "Upload" and upload your dataset ZIP file
   - Note the dataset name (e.g., "plant-dataset")

3. **Set Up the Notebook**
   - In your Kaggle notebook, make sure GPU is enabled (Settings → Accelerator → GPU)
   - Upload the files from this directory to your Kaggle notebook

4. **Run the Training**
   ```python
   !python kaggle_train.py
   ```

5. **Download the Model**
   After training completes, the model will be saved to:
   - `/kaggle/working/plant_disease_model.pth`
   - `/kaggle/working/class_indices.json`

   You can download these files using:
   ```python
   from IPython.display import FileLink
   FileLink('plant_disease_model.pth')
   FileLink('class_indices.json')
   ```

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- tqdm
- numpy
- PIL

All required packages are pre-installed on Kaggle. If you need to install additional packages, use:
```python
!pip install -r requirements.txt
```

## Expected Training Time on Kaggle GPU
- ~5-10 minutes per epoch
- ~2-3 hours for 20 epochs

## Notes
- The script automatically saves the best model based on validation accuracy
- Learning rate is reduced when validation loss plateaus
- Early stopping is implemented if no improvement is seen for 5 epochs
