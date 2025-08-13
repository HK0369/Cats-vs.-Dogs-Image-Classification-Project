import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Cat vs. Dog Classifier",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- Model Loading ---
# Use st.cache_resource to load the model only once, improving performance.
@st.cache_resource
def load_keras_model():
    """
    Loads the pre-trained Keras model from the specified path.
    Handles potential FileNotFoundError.
    """
    # Using the absolute path you provided.
    model_path = r"C:\Users\nhari\OneDrive\Desktop\cats_vs_dogs_project\models\cats_vs_dogs_model.h5"
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found at '{model_path}'. Please ensure the path is correct.")
        st.stop()
        
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        st.stop()

# --- Prediction Function ---
def predict(image_to_predict, model):
    """
    Takes a PIL image and a loaded Keras model, preprocesses the image,
    and returns the model's prediction score.

    Args:
        image_to_predict (PIL.Image.Image): The image to classify.
        model (tf.keras.Model): The loaded classification model.

    Returns:
        float: The prediction score from the model.
    """
    # Preprocess the image to match the model's input requirements
    img = image_to_predict.resize((150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Rescale pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch

    # Make prediction
    prediction_score = model.predict(img_array)[0][0]
    return prediction_score

# --- Streamlit App UI ---

# Title and description
st.title("🐾 Cat vs. Dog Image Classifier")
st.markdown(
    "Upload an image of a cat or a dog, and this app will now use your trained "
    "Convolutional Neural Network (CNN) to predict which one it is."
)

# Load the model
model = load_keras_model()

# Sidebar for additional information
st.sidebar.header("About the App")
st.sidebar.info(
    "This web application demonstrates a deep learning model in action. "
    "The model was trained on thousands of images of cats and dogs to learn "
    "the distinguishing features of each animal."
)
st.sidebar.success("Project by: **Hari Krishna N**")


# File uploader widget
uploaded_file = st.file_uploader(
    "Choose an image file", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    
    # Center the image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # --- FIXED: use_container_width is the new parameter ---
        st.image(image, caption='Uploaded Image', use_container_width=True)

    # Add a spinner while classifying
    with st.spinner('Analyzing the image...'):
        # Get the prediction
        score = predict(image, model)
        
        # Calculate confidence
        confidence = score if score > 0.5 else 1 - score

        # Display the result
        st.subheader("Prediction:")
        if score > 0.5:
            st.success(f"**This looks like a Dog!** �")
        else:
            st.success(f"**This looks like a Cat!** 🐈")

        st.info(f"**Confidence:** {confidence * 100:.2f}%")

else:
    st.info("Please upload an image file to get a prediction.")

