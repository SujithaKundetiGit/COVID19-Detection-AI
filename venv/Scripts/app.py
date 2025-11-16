import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io

# --- Configuration ---
MODEL_PATH = 'D:\VisualStudioProjGuvi\Covid19project\covid_detection_model.keras' # NEW Keras native format
IMG_HEIGHT = 224 
IMG_WIDTH = 224

CLASS_NAMES = ['COVID-19', 'Normal', 'Viral Pneumonia'] 

# --- Load Model (Caching improves performance) ---
@st.cache_resource
def load_model():
    """Loads the trained Keras model."""
    try:
        # Load the model from the new .keras file path
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Ensure you have run the training script and the model is saved as 'covid_detection_model.keras'.")
        st.stop() 

# Load the model once
model = load_model()

# --- Prediction Function ---
def predict_image(image_file, model):
    """
    Preprocesses the image, makes a prediction, and returns the results.
    """
    # 1. Image Preprocessing (Resize and Normalize)
    image = Image.open(image_file).convert('RGB')
    image = image.resize((IMG_HEIGHT, IMG_WIDTH))
    
    # Convert image to numpy array and normalize pixels (0-255 -> 0-1)
    img_array = np.array(image) / 255.0
    
    # Expand dimensions to create a batch size of 1 
    img_array = np.expand_dims(img_array, axis=0)

    # 2. Prediction
    predictions = model.predict(img_array)
    
    # 3. Process Results
    score = tf.nn.softmax(predictions[0])
    predicted_class_index = np.argmax(score)
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = np.max(score)

    return predicted_class, confidence, predictions[0]

# --- Streamlit Application Layout ---
st.set_page_config(page_title="COVID-19 Chest X-ray Diagnostic Assistant", layout="centered")

st.title("🫁 Multi-class COVID-19 Detection from Chest X-ray Images")
st.markdown("---")
st.subheader("Clinical Support and Remote Healthcare Assistant")
st.write("Upload a chest X-ray image to classify it as **COVID-19**, **Viral Pneumonia**, or **Normal**.")

uploaded_file = st.file_uploader(
    "Choose a Chest X-ray Image (.png, .jpg, .jpeg)", 
    type=['png', 'jpg', 'jpeg']
)

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.markdown("---")
    
    # Run prediction upon button click
    if st.button('Analyze X-ray'):
        with st.spinner('Analyzing image...'):
            try:
                # Get the prediction
                predicted_class, confidence, raw_predictions = predict_image(uploaded_file, model)
                
                st.success("✅ Analysis Complete!")

                # --- Display Results ---
                st.markdown(f"### Predicted Condition: **{predicted_class}**")
                st.markdown(f"Confidence: **{confidence * 100:.2f}%**")
                
                # Highlight COVID-19 result for immediate attention
                if predicted_class == 'COVID-19':
                    st.warning("🚨 ALERT: High suspicion of COVID-19. Further clinical evaluation is mandatory.")
                
                # Detailed Confidence Breakdown (Optional but informative)
                st.markdown("#### Confidence Breakdown")
                
                # Format predictions for display
                results_df = {
                    'Condition': CLASS_NAMES,
                    'Probability': [f"{p * 100:.2f}%" for p in raw_predictions]
                }
                st.table(results_df)

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

st.markdown("---")
