import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import json
import numpy as np

# Define the model architecture (same as used in training)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=38):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        self.flattened_size = 128 * 16 * 16
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Load class indices
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
    # Convert string keys to integers
    class_indices = {int(k): v for k, v in class_indices.items()}
    idx_to_class = {v: k for k, v in class_indices.items()}

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN(num_classes=len(class_indices)).to(device)
model.load_state_dict(torch.load('plant_disease_model.pth', map_location=device))
model.eval()

# Image transformations
image_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image):
    """Predict the class of an image"""
    # Preprocess
    image = image_transforms(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
    # Get top 3 predictions
    top3_prob, top3_catid = torch.topk(probabilities, 3)
    predictions = []
    for i in range(top3_prob.size(0)):
        class_idx = top3_catid[i].item()
        class_name = class_indices.get(class_idx, f"Class {class_idx}")
        # Convert to human-readable format
        class_name = class_name.replace('___', ' ').replace('_', ' ').title()
        predictions.append({
            'class': class_name,
            'probability': f"{top3_prob[i].item() * 100:.2f}%"
        })
    
    return predictions

# Page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar with additional information
with st.sidebar:
    st.title("🌿 Plant Disease Classifier")
    st.write("---")
    
    # About section
    st.subheader("About")
    st.markdown("""
    This app helps identify plant diseases using deep learning. 
    Upload an image of a plant leaf, and the model will predict 
    the most likely disease affecting it.
    """)
    
    # Supported diseases section
    st.subheader("Supported Diseases")
    if st.checkbox("Show supported diseases"):
        try:
            with open('class_indices.json', 'r') as f:
                diseases = json.load(f).values()
                # Format disease names for better readability
                formatted_diseases = [d.replace('___', ' ').replace('_', ' ').title() 
                                    for d in diseases]
                formatted_diseases.sort()  # Sort alphabetically
                
                # Display in two columns for better layout
                col1, col2 = st.columns(2)
                half = (len(formatted_diseases) + 1) // 2
                
                with col1:
                    for disease in formatted_diseases[:half]:
                        st.markdown(f"- {disease}")
                with col2:
                    for disease in formatted_diseases[half:]:
                        st.markdown(f"- {disease}")
        except Exception as e:
            st.error("Could not load disease list.")
    
    st.write("---")
    st.markdown("*Upload an image to get started!*")

# Main content
st.title("🌱 Plant Disease Classifier")
st.write("Upload an image of a plant leaf to detect potential diseases")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Make prediction
        with st.spinner('Analyzing...'):
            predictions = predict(image)
        
        # Display results
        st.subheader("Top Predictions:")
        for i, pred in enumerate(predictions, 1):
            st.write(f"{i}. {pred['class']} - {pred['probability']}")
        
        # Show confidence level
        confidence = float(predictions[0]['probability'].strip('%'))
        if confidence > 80:
            st.success("✅ High confidence in prediction!")
        elif confidence > 50:
            st.warning("⚠️ Moderate confidence in prediction.")
        else:
            st.info("ℹ️ Low confidence in prediction. Please ensure the image is clear and shows a plant leaf.")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please try with a different image or make sure the image is clear and shows a plant leaf.")

# Add some styling
st.markdown("""
    <style>
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
        padding: 1rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stMarkdown h2 {
        color: #2e7d32;
    }
    .stMarkdown h3 {
        color: #388e3c;
    }
    </style>
""", unsafe_allow_html=True)