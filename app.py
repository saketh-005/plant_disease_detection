import os
from io import BytesIO
# Configure Matplotlib to use a non-interactive backend
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend which doesn't require a display
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

# Create the directory if it doesn't exist
try:
    os.makedirs('/tmp/matplotlib', exist_ok=True)
    os.chmod('/tmp/matplotlib', 0o777)
except Exception as e:
    pass  # Directory creation is not critical

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
import torch
from torchvision import transforms, models
import torch.nn as nn
import json
from pathlib import Path
import time

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Set device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Set page config
st.set_page_config(
    page_title="Plant Disease Prediction",
    page_icon="🌱",
    layout="wide"
)

# Title and description
st.title("🌿 Plant Disease Prediction")
st.write("Upload an image of a plant leaf to detect potential diseases.")

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.write("This app uses a deep learning model to detect plant diseases from leaf images.")
    st.write("### How to use:")
    st.write("1. Upload an image of a plant leaf")
    st.write("2. The app will process the image")
    st.write("3. View the prediction results")
    st.write("\nNote: This is a demo application. For production use, please train on a larger dataset.")

# Constants
MODEL_PATH = 'plant_disease_model.h5'
IMG_SIZE = 224

# Class mapping for PlantVillage dataset
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Disease information
DISEASE_INFO = {
    'Apple___Apple_scab': {
        'symptoms': 'Olive-green to black, circular spots on leaves that may become raised and velvety.',
        'treatment': 'Apply fungicides in early spring and remove fallen leaves in autumn.'
    },
    'Tomato___Early_blight': {
        'symptoms': 'Dark, concentric spots on lower leaves that may develop a target-like appearance.',
        'treatment': 'Use fungicides, remove infected leaves, and ensure good air circulation.'
    },
    'default': {
        'symptoms': 'Consult with a plant pathologist for accurate diagnosis.',
        'treatment': 'Isolate the plant and consult with a local agricultural extension service.'
    }
}

# Default class indices (fallback if file not found)
DEFAULT_CLASS_INDICES = {
    'Pepper__bell___Bacterial_spot': 0,
    'Pepper__bell___healthy': 1,
    'Potato___Early_blight': 2,
    'Potato___Late_blight': 3,
    'Potato___healthy': 4,
    'Tomato_Bacterial_spot': 5,
    'Tomato_Early_blight': 6,
    'Tomato_Late_blight': 7,
    'Tomato_Leaf_Mold': 8,
    'Tomato_Septoria_leaf_spot': 9,
    'Tomato_Spider_mites_Two_spotted_spider_mite': 10,
    'Tomato__Target_Spot': 11,
    'Tomato__Tomato_YellowLeaf__Curl_Virus': 12,
    'Tomato__Tomato_mosaic_virus': 13,
    'Tomato_healthy': 14
}

# Load the model and class indices
@st.cache_resource
def load_model():
    try:
        # Try to load class indices from file, fall back to default if not found
        try:
            with open('class_indices.json', 'r') as f:
                class_indices = json.load(f)
            print("Loaded class indices from file")
        except FileNotFoundError:
            class_indices = DEFAULT_CLASS_INDICES
            print("Using default class indices")
        
        # Create model
        print("Creating model...")
        model = models.resnet18(weights=None)  # We'll load our own weights
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(class_indices))
        
        # Load weights if available
        try:
            checkpoint = torch.load('plant_disease_model.pth', map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded model weights")
        except FileNotFoundError:
            print("Warning: Model weights not found. Using random weights.")
        
        model = model.to(device)
        model.eval()
        
        # Reverse the dictionary to get index to class name mapping
        idx_to_class = {v: k for k, v in class_indices.items()}
        print(f"Model loaded with {len(idx_to_class)} classes")
        return model, idx_to_class
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error loading model: {error_details}")
        st.error(f"Error loading model: {e}")
        return None, None

model, idx_to_class = load_model()

def preprocess_image(img):
    """Preprocess the image for prediction"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0).to(device)  # Add batch dimension and move to device

def predict_disease(image):
    if model is None or idx_to_class is None:
        return "Model not loaded", 0.0
    
    try:
        # Preprocess the image
        input_tensor = preprocess_image(image)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)
            predicted_class = idx_to_class.get(str(predicted_idx.item()), "Unknown")
            
        return predicted_class, confidence.item()
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return "Prediction error", 0.0

def random_predictions():
    """Generate random predictions for demo purposes."""
    np.random.seed(42)
    indices = np.random.choice(len(CLASS_NAMES), 3, replace=False)
    scores = np.random.dirichlet(np.ones(3), size=1)[0]
    return list(zip([CLASS_NAMES[i] for i in indices], scores)), 0.1

def display_disease_info(disease_name):
    """Display detailed information about the predicted disease."""
    # Get disease information or use default
    info = DISEASE_INFO.get(disease_name, DISEASE_INFO['default'])
    
    st.markdown("### Disease Information")
    st.markdown(f"**{disease_name.replace('_', ' ').title()}**")
    
    # Display symptoms and treatment
    with st.expander("ℹ️ Symptoms"):
        st.write(info['symptoms'])
    
    with st.expander("💊 Treatment"):
        st.write(info['treatment'])
    
    # Add prevention tips
    with st.expander("🛡️ Prevention Tips"):
        st.write("""
        - Ensure proper spacing between plants for good air circulation
        - Water at the base of plants to keep foliage dry
        - Rotate crops to prevent disease buildup in soil
        - Remove and destroy infected plant material
        - Use disease-resistant varieties when available
        """)

def main():
    # Load the model (cached for performance)
    model, idx_to_class = load_model()
    
    # Check if model loaded successfully
    if model is None or idx_to_class is None:
        st.error("⚠️ Failed to load the model. Some features may not work correctly.")
        st.info("Please check if the model files are present in the correct location.")
    
    # Sidebar options
    with st.sidebar.expander("⚙️ Settings"):
        confidence_threshold = st.slider(
            "Minimum Confidence Threshold", 
            min_value=0.1, 
            max_value=0.9, 
            value=0.5, 
            step=0.1,
            help="Adjust the minimum confidence level for predictions"
        )
    
    # Main content
    st.markdown("## 🌱 Plant Disease Detection")
    st.markdown("Upload an image of a plant leaf to detect potential diseases.")
    
    # File uploader
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # If no file is uploaded or selected, show a message
    if uploaded_file is None and 'uploaded_file' not in st.session_state:
        st.info("👆 Upload an image to get started")
    
    if uploaded_file is not None:
        try:
            # Reset the file pointer to the beginning
            if hasattr(uploaded_file, 'seek'):
                uploaded_file.seek(0)
                
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with st.spinner("🔍 Analyzing the image..."):
                # Preprocess the image
                processed_image = preprocess_image(image)
                
                # Get predictions
                predictions, inference_time = predict_with_model(model, processed_image, top_k=3)
                
                # Filter predictions by confidence threshold
                filtered_predictions = [(d, s) for d, s in predictions if s >= confidence_threshold]
                
                if not filtered_predictions:
                    st.warning("No predictions met the confidence threshold. Try adjusting the threshold or upload a clearer image.")
                else:
                    # Display results in two columns
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("### 📊 Prediction Results")
                        
                        # Create a DataFrame for the predictions
                        df = pd.DataFrame({
                            'Disease': [p[0].replace('_', ' ').title() for p in filtered_predictions],
                            'Confidence': [p[1] for p in filtered_predictions]
                        })
                        
                        # Display as a bar chart
                        fig = px.bar(
                            df, 
                            x='Confidence', 
                            y='Disease',
                            orientation='h',
                            title='Prediction Confidence',
                            labels={'Confidence': 'Confidence Score', 'Disease': 'Disease'},
                            color='Confidence',
                            color_continuous_scale='Viridis'
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display inference time
                        st.caption(f"⚡ Inference time: {inference_time*1000:.1f}ms")
                    
                    with col2:
                        # Display detailed information about the top prediction
                        top_disease = filtered_predictions[0][0]
                        display_disease_info(top_disease)
                    
                    # Add a section for user feedback
                    st.markdown("---")
                    with st.expander("📝 Provide Feedback"):
                        st.write("Help us improve our model!")
                        feedback = st.radio(
                            "Was this prediction accurate?",
                            ("Yes", "Partially", "No")
                        )
                        if st.button("Submit Feedback"):
                            # In a real app, you would save this feedback
                            st.success("Thank you for your feedback!")
            
            # Add some space at the bottom
            st.markdown("---")
            st.markdown("""
            ### ℹ️ About This Tool
            This application uses deep learning to identify plant diseases from leaf images. 
            The model has been trained on the [PlantVillage dataset](https://plantvillage.psu.edu/) 
            and can detect various plant diseases.
            
            **Note:** This is a demonstration application. For real-world use, consult with 
            agricultural experts and use laboratory testing for accurate disease diagnosis.
            """)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please try another image or check the console for details.")
            
    # Add a footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray;">
        <p>🌿 Plant Disease Detection App • Built with Streamlit • 
        <a href="#" style="color: gray;">Terms of Use</a> • 
        <a href="#" style="color: gray;">Privacy Policy</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
