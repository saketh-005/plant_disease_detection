# 🌿 Plant Disease Detection

A deep learning-based web application that identifies plant diseases from leaf images using a custom CNN model, providing fast and accurate predictions.

## 🚀 Features

- 🌱 **Multi-class Classification**: Identifies 38 different plant diseases
- 🖼️ **Image Upload**: Simple drag-and-drop interface for uploading plant leaf images
- 🎯 **High Accuracy**: Powered by a custom CNN model trained on PlantVillage dataset
- 📊 **Confidence Scores**: Displays prediction confidence levels
- 📱 **Responsive Design**: Works on both desktop and mobile devices
- 🚀 **Fast Inference**: Optimized for quick predictions
- 🧠 **Train Your Own Model**: Includes training scripts for custom model development

## 🛠️ Prerequisites

- Python 3.8+
- pip (Python package manager)
- Git (for cloning the repository)
- (For training) NVIDIA GPU with CUDA support (recommended)

## 🚀 Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/saketh-005/plant_disease_detection.git
   cd plant_disease_detection
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

5. Open your browser and navigate to `http://localhost:8501`

## 🧠 Training Your Own Model

The `kaggle_plant_disease` directory contains everything you need to train your own plant disease classification model.

### Prerequisites for Training
- Kaggle account (for GPU access)
- PlantVillage dataset (or your own dataset)
- Basic knowledge of PyTorch

### Training on Kaggle (Recommended)

1. **Prepare Your Dataset**
   - Download the PlantVillage dataset or prepare your own dataset with the following structure:
     ```
     dataset/
     ├── train/
     │   ├── class1/
     │   ├── class2/
     │   └── ...
     └── valid/
         ├── class1/
         ├── class2/
         └── ...
     ```

2. **Upload to Kaggle**
   - Compress your dataset folder into a ZIP file
   - Upload it to Kaggle as a new dataset
   - Create a new Kaggle Notebook with GPU acceleration enabled

3. **Run the Training**
   - Upload the training script: `kaggle_plant_disease/kaggle_train.py`
   - Update the dataset path in the script if needed
   - Run the training:
     ```python
     !python kaggle_train.py
     ```

4. **Download the Model**
   - After training, download the model files:
     - `plant_disease_model.pth` (model weights)
     - `class_indices.json` (class labels mapping)

### Training Parameters
You can customize the training by modifying these parameters in `kaggle_train.py`:
- `BATCH_SIZE`: Number of samples per batch (default: 64)
- `EPOCHS`: Number of training epochs (default: 20)
- `LEARNING_RATE`: Learning rate for the optimizer (default: 0.001)
- `IMG_SIZE`: Input image size (default: 256)

## 🏗️ Project Structure

```
plant-disease-detection/
├── app.py                # Main Streamlit application
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
├── Dockerfile           # Docker configuration
├── .gitignore          # Git ignore file
├── kaggle_plant_disease/ # Training scripts and utilities
│   ├── kaggle_train.py  # Model training script
│   ├── README.md       # Training instructions
│   └── requirements.txt # Training dependencies
├── plant_disease_model.pth  # Pre-trained model weights
└── class_indices.json   # Class labels mapping
```

## 🌐 Deployment

### Local Deployment

#### Using Docker
```bash
# Build the Docker image
docker build -t plant-disease-detection .

# Run the container
docker run -p 8501:8501 plant-disease-detection
```

### Cloud Deployment

#### Hugging Face Spaces
1. Fork this repository
2. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
3. Click "Create new Space"
4. Select "Docker" as the SDK
5. Configure the Space with your repository details

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Note

This is a demonstration application. For production use, please consult with agricultural experts and use laboratory testing for accurate disease diagnosis.

## 📄 License

Copyright © 2025 [Saketh Jangala](https://github.com/saketh-005)

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
