import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2 # OpenCV is needed for image manipulation (like resizing and blending)
import matplotlib.pyplot as plt
import io

# --- Configuration ---
MODEL_PATH = r'D:\VisualStudioProjGuvi\Covid19project\covid_detection_model.keras'
IMG_HEIGHT = 224 
IMG_WIDTH = 224
# NOTE: Verify this order matches your training script's class_indices
CLASS_NAMES = ['COVID-19', 'Normal', 'Viral Pneumonia'] 
LAST_CONV_LAYER_NAME = 'conv5_block3_out' # Common last conv layer for ResNet-50

# --- Load Model (Caching improves performance) ---
@st.cache_resource
def load_model():
    """Loads the trained Keras model."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Ensure you have run the training script and the model is saved as 'covid_detection_model.keras'.")
        st.stop() 

model = load_model()

# --- Grad-CAM Functions ---

def get_gradcam_model(model, last_conv_layer_name):
    """Creates a Keras Model that maps input image to the activations of the 
       last convolutional layer and the final output predictions."""
    
    # 1. Model that outputs the last convolutional layer activations
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    return grad_model

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Computes the Grad-CAM heatmap for a given image array and model."""
    
    grad_model = get_gradcam_model(model, last_conv_layer_name)

    # Watch the image tensor
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        
        # If no specific index is provided, use the highest prediction score
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        
        # Get the loss (score) of the predicted class
        class_channel = preds[:, pred_index]

    # Compute gradients of the predicted class score with respect to the output 
    # of the last convolutional layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Pool the gradients over all the spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map by "how important" this channel is
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap to range [0, 1]
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def display_gradcam(img, heatmap):
    """Superimposes the heatmap onto the original image and returns the result."""
    
    # Rescale heatmap to a size that matches the image dimensions (224x224)
    heatmap = np.uint8(255 * heatmap)
    
    # Use cv2 to resize the heatmap to the image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Apply the 'jet' color map (or 'viridis', 'hot')
    cmap = plt.cm.get_cmap('jet')
    colors = cmap(heatmap)
    
    # Convert RGBA to RGB and scale to 255
    colors = np.delete(colors, 3, 2)
    colors = (colors * 255).astype(np.uint8)

    # Convert the input image back to 8-bit RGB if it was normalized
    # Rescale to 255 for blending if it was passed as 0-1 float array
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    # 0.4 controls the opacity of the heatmap
    superimposed_img = cv2.addWeighted(img, 0.6, colors, 0.4, 0)

    return superimposed_img


# --- Prediction Function (Updated to use Image object directly) ---
def predict_image(image_file, model):
    """
    Preprocesses the image, makes a prediction, and returns the results.
    """
    image = Image.open(image_file).convert('RGB')
    
    # 1. Preprocessing for Model Input
    processed_image = image.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = np.array(processed_image) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    # 2. Prediction
    predictions = model.predict(img_input)
    score = tf.nn.softmax(predictions[0])
    predicted_class_index = np.argmax(score)
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = np.max(score)

    return predicted_class, confidence, predictions[0], img_array, predicted_class_index

# --- Streamlit Application Layout ---
st.set_page_config(page_title="COVID-19 Chest X-ray Diagnostic Assistant", layout="wide")

st.title("🫁 Multi-class COVID-19 Detection from Chest X-ray Images")
st.markdown("---")
st.subheader("Clinical Support and Remote Healthcare Assistant (with Explainable AI)")
st.write("Upload a chest X-ray image to classify it as **COVID-19**, **Viral Pneumonia**, or **Normal**.")

uploaded_file = st.file_uploader(
    "Choose a Chest X-ray Image (.png, .jpg, .jpeg)", 
    type=['png', 'jpg', 'jpeg']
)

if uploaded_file is not None:
    
    st.markdown("---")
    
    # Run prediction upon button click
    if st.button('Analyze X-ray and Generate Explainability Map'):
        with st.spinner('Analyzing image and generating Grad-CAM visualization...'):
            try:
                # 1. Get the prediction and processed image array
                predicted_class, confidence, raw_predictions, img_array, pred_index = predict_image(uploaded_file, model)
                
                # 2. Calculate Grad-CAM Heatmap
                # We need to reshape the img_array back to (1, 224, 224, 3) for the Grad-CAM function
                heatmap = make_gradcam_heatmap(
                    np.expand_dims(img_array, axis=0), 
                    model, 
                    LAST_CONV_LAYER_NAME, 
                    pred_index=pred_index
                )
                
                # 3. Generate Superimposed Image
                gradcam_image = display_gradcam(img_array, heatmap)

                st.success("✅ Analysis Complete!")
                
                # --- Display Results and Visualization ---
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"### Predicted Condition: **{predicted_class}**")
                    st.markdown(f"Confidence: **{confidence * 100:.2f}%**")
                    
                    if predicted_class == 'COVID-19':
                        st.warning("🚨 ALERT: High suspicion of COVID-19. Further clinical evaluation is mandatory.")
                    
                    st.markdown("#### Confidence Breakdown")
                    results_df = {
                        'Condition': CLASS_NAMES,
                        'Probability': [f"{p * 100:.2f}%" for p in raw_predictions]
                    }
                    st.table(results_df)

                with col2:
                    st.markdown("### Grad-CAM Visualization")
                    st.image(gradcam_image, caption='Region of Interest for Prediction', use_column_width=True)
                    st.markdown("""
                        <p style='font-size: small; color: gray;'>
                        The heatmap highlights the areas (yellow/red) of the X-ray image that were most influential
                        in the model's decision for the predicted class.
                        </p>
                    """, unsafe_allow_html=True)


            except Exception as e:
                st.error(f"An error occurred during prediction or Grad-CAM generation: {e}")
                st.exception(e) # Display the full error trace for debugging

st.markdown("---")
