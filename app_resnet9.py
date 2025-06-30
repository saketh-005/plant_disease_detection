import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import json
import numpy as np
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        max-width: 1000px;
        padding: 2rem;
    }
    .title {
        text-align: center;
        color: #2e8b57;
    }
    .prediction {
        font-size: 1.2rem;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .healthy {
        background-color: #d4edda;
        color: #155724;
    }
    .diseased {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Model class (same as in training)
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.conv(x)

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        # First conv block
        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Conv2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Res1
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Conv3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Res2
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load class indices
@st.cache_data
def load_class_indices():
    with open('class_indices.json', 'r') as f:
        return json.load(f)

# Load model
@st.cache_resource
def load_model():
    class_indices = load_class_indices()
    model = ResNet9(3, len(class_indices))
    model.load_state_dict(torch.load('plant_disease_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocess image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Predict function
def predict(image, model, class_indices):
    idx_to_class = {int(k): v for k, v in class_indices.items()}
    
    # Preprocess
    input_tensor = preprocess_image(image)
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        predicted_class = idx_to_class[predicted_idx.item()]
        
    return predicted_class, confidence.item()

# Main app
def main():
    st.title("🌱 Plant Disease Classifier")
    st.markdown("---")
    
    # Load model and class indices
    try:
        model = load_model()
        class_indices = load_class_indices()
        idx_to_class = {int(k): v for k, v in class_indices.items()}
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please make sure you have trained the model first by running 'python resnet9_train.py'")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an image of a plant leaf", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Make prediction
        with st.spinner('Analyzing...'):
            predicted_class, confidence = predict(image, model, class_indices)
            
            # Display result
            plant, status = predicted_class.split('___')
            is_healthy = status == 'healthy'
            
            st.markdown("### Prediction Result")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Plant", plant.replace('_', ' ').title())
            with col2:
                status_display = "Healthy 🟢" if is_healthy else "Diseased 🔴"
                st.metric("Status", status_display)
            
            if not is_healthy:
                st.metric("Disease", status.replace('_', ' ').title())
            
            st.metric("Confidence", f"{confidence*100:.2f}%")
            
            # Show info based on prediction
            if is_healthy:
                st.success(f"This {plant.replace('_', ' ').lower()} leaf appears to be healthy!")
            else:
                st.warning(f"This {plant.replace('_', ' ').lower()} leaf shows signs of {status.replace('_', ' ').lower()}.")
                
                # Add some general advice (you can expand this)
                st.info("""
                **Recommendations:**
                - Isolate the affected plant to prevent spread
                - Remove severely infected leaves
                - Consider using appropriate fungicides/pesticides
                - Ensure proper spacing and air circulation
                - Maintain optimal watering practices
                """)
    else:
        st.info("Please upload an image of a plant leaf to check for diseases.")
    
    # Add some information about the model
    st.markdown("---")
    st.markdown("""
    ### About this App
    This app uses a ResNet9 deep learning model to identify plant diseases from leaf images. 
    It can detect 38 different classes of plant diseases across 14 plant species.
    
    **How to use:**
    1. Upload an image of a plant leaf
    2. The model will analyze the image
    3. View the prediction and recommendations
    
    **Note:** For best results, use clear, well-lit photos of individual leaves.
    """)

if __name__ == "__main__":
    main()
