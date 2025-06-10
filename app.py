import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf

# Dice Coefficient (custom metric)
def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

# Jaccard Index (custom metric)
def jaccard_index(y_true, y_pred, smooth=1):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# Load your trained U-Net model (use .keras extension)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "best_model.keras", 
        custom_objects={"dice_coefficient": dice_coefficient, "jaccard_index": jaccard_index}
    )

# Preprocess the uploaded image for prediction
def preprocess_image(img, size=(512, 512)):
    img = img.resize(size).convert("L")  # Resize to 512x512 and convert to grayscale
    img_array = np.array(img) / 255.0  # Normalize the pixel values
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Postprocess the predicted mask
def postprocess_mask(mask, original_size):
    mask = mask[0, :, :, 0]  # Remove batch and channel dimensions
    mask = (mask > 0.5).astype(np.uint8)  # Binarize the mask
    mask = cv2.resize(mask, original_size[::-1])  # Resize to the original image size
    return mask

# Title and instructions
st.title("Lung Segmentation")
st.write("""
Upload a lung scan image to perform segmentation. 
The model will generate a segmented mask for the lungs.
""")

# File uploader
uploaded_file = st.file_uploader("Upload a lung scan image (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Load and display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    original_size = img.size  # Save the original size for resizing the mask
    img_array = preprocess_image(img)

    # Load the model
    model = load_model()

    # Perform segmentation
    st.write("Processing the image...")
    prediction = model.predict(img_array)
    segmented_mask = postprocess_mask(prediction, original_size)

    # Display the segmented mask
    st.image(segmented_mask * 255, caption="Segmented Mask", use_column_width=True, clamp=True)

    # Allow the user to download the segmented mask
    mask_image = Image.fromarray((segmented_mask * 255).astype(np.uint8))
    st.download_button(
        "Download Segmented Mask",
        data=mask_image.tobytes(),
        file_name="segmented_mask.png",
        mime="image/png"
    )
