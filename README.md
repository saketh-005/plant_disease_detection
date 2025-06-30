# 🌿 Plant Disease Detection

A deep learning-based web application that identifies plant diseases from leaf images using a custom CNN model, providing fast and accurate predictions.

## 🚀 Features

- 🌱 Identify 38 different plant diseases using a custom CNN model
- 📊 Interactive prediction visualization with confidence scores
- 📱 Mobile-responsive design
- ⚡ Fast inference with model caching
- 📝 Detailed disease information and treatment suggestions
- 🐳 Docker container support for easy deployment
- 🌐 Deployable on Hugging Face Spaces

## 🛠️ Prerequisites

- Python 3.8+
- pip
- Git

## 🚀 Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/saketh-005/plant_disease_detection.git
   cd plant_disease_detection
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
   The app will be available at `http://localhost:8501`

## 🐳 Docker Deployment

Build and run using Docker:
```bash
docker build -t plant-disease-detection .
docker run -p 8501:7860 plant-disease-detection
```

## 🌐 Online Demo

Try the live demo on [Hugging Face Spaces](https://huggingface.co/spaces/saketh-005/plant-disease-detection).

## 🛠️ Project Structure

```
.
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration
├── space.yml              # Hugging Face Spaces configuration
├── class_indices.json     # Class labels mapping
├── .gitignore            # Git ignore file
└── README.md             # This file
```

## 📝 Note

This is a demonstration application. For production use, please consult with agricultural experts and use laboratory testing for accurate disease diagnosis.

## 📄 License

Copyright © 2025 [Saketh Jangala](https://github.com/saketh-005)

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
