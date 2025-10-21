import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load trained model
@st.cache_resource
def load_model():
    # Get current working directory for debugging
    current_dir = os.getcwd()
    st.sidebar.write(f"Current directory: {current_dir}")
    
    # List all files in current directory for debugging
    try:
        files_in_dir = os.listdir('.')
        st.sidebar.write(f"Files in current directory: {files_in_dir}")
    except:
        st.sidebar.write("Could not list files in current directory")
    
    # Try different possible paths for the model
    possible_paths = [
        'mnist_model.h5',  # Local path when running from streamlit_app directory
        './mnist_model.h5',  # Current directory
        'task2_deep_learning/streamlit_app/mnist_model.h5',  # Streamlit Cloud path
        '../mnist_model.h5',  # Alternative local path
        os.path.join(os.getcwd(), 'mnist_model.h5'),  # Absolute path
        os.path.join(os.path.dirname(__file__), 'mnist_model.h5')  # Same directory as script
    ]
    
    st.sidebar.write("Trying to find model file...")
    
    for model_path in possible_paths:
        st.sidebar.write(f"Checking: {model_path}")
        if os.path.exists(model_path):
            try:
                st.sidebar.write(f"Found model at: {model_path}")
                model = tf.keras.models.load_model(model_path, compile=False)
                # Display model info for debugging
                st.sidebar.success(f"Model loaded from: {model_path}")
                st.sidebar.write(f"Model input shape: {model.input_shape}")
                st.sidebar.write(f"Model output shape: {model.output_shape}")
                return model
            except Exception as e:
                st.sidebar.error(f"Error loading model from {model_path}: {str(e)}")
                continue
        else:
            st.sidebar.write(f"Not found: {model_path}")
    
    # If no path works, show error with all attempted paths
    st.error(f"Model file not found. Tried paths: {possible_paths}")
    st.error("Please ensure the mnist_model.h5 file is in the same directory as app.py")
    return None

model = load_model()

def preprocess_image(image, model_input_shape):
    """
    Preprocess image to match model input requirements
    """
    # Convert to grayscale and resize to 28x28
    if image.mode != 'L':
        image = image.convert('L')
    image = image.resize((28, 28))
    
    # Convert to numpy array and normalize
    img_array = np.array(image) / 255.0
    
    # Check if we need to invert the image (MNIST has white digits on black background)
    # If the image has mostly white pixels, invert it
    if np.mean(img_array) > 0.5:
        img_array = 1.0 - img_array
    
    # Handle different model input shapes
    if len(model_input_shape) == 3:  # (28, 28, 1) - CNN model
        img_array = img_array.reshape(1, 28, 28, 1)
    elif len(model_input_shape) == 2:  # (784,) - Dense model
        img_array = img_array.reshape(1, 784)
    else:
        # Default to CNN format
        img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array, image

st.title("MNIST Digit Classifier")
st.markdown("Upload a handwritten digit image to get an AI-powered prediction!")

if model is None:
    st.error("Failed to load the model. Please check the deployment logs.")
    st.stop()

# Add instructions
with st.expander("📋 Instructions for best results"):
    st.markdown("""
    **For accurate predictions, please follow these guidelines:**
    
    1. **Image Quality**: Use clear, well-lit images
    2. **Digit Style**: Write digits similar to how they appear in MNIST dataset
    3. **Background**: White or light background works best
    4. **Size**: Any size is fine - the app will resize to 28x28 pixels
    5. **Format**: PNG or JPG files are supported
    
    **Tips:**
    - The app automatically inverts images if needed (MNIST uses white digits on black background)
    - If you get low confidence, try a clearer image or different digit style
    - The model works best with single, centered digits
    """)

uploaded_file = st.file_uploader("Upload a handwritten digit image", type=["png", "jpg"])

if uploaded_file is not None:
    try:
        # Load the image
        original_image = Image.open(uploaded_file)
        
        # Preprocess the image using our function
        img_array, processed_image = preprocess_image(original_image, model.input_shape[1:])

        # Display the processed image
        st.image(processed_image, caption="Processed Image (28x28)", use_column_width=True)
        
        # Show the processed array for debugging
        with st.expander("Image Processing Details"):
            st.write("Original image mean:", np.mean(np.array(original_image.convert('L').resize((28, 28))) / 255.0))
            st.write("Processed image mean:", np.mean(img_array))
            st.write("Image shape:", img_array.shape)
            st.write("Image min/max:", img_array.min(), "/", img_array.max())
            st.write("Model input shape:", model.input_shape)

        # Make prediction
        prediction = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"🎯 Predicted Digit: {predicted_class}")
        with col2:
            st.subheader(f"📊 Confidence: {confidence:.2f}%")
        
        # Show detailed prediction scores
        st.write("**Prediction scores for each digit:**")
        prediction_scores = {str(i): float(prediction[0][i]) for i in range(10)}
        st.bar_chart(prediction_scores)
        
        # Show top 3 predictions
        top3_indices = np.argsort(prediction[0])[-3:][::-1]
        st.write("**Top 3 predictions:**")
        for i, idx in enumerate(top3_indices):
            confidence_score = prediction[0][idx] * 100
            st.write(f"{i+1}. **Digit {idx}**: {confidence_score:.2f}%")
            
        # Add some styling based on confidence
        if confidence > 90:
            st.success("✅ High confidence prediction!")
        elif confidence > 70:
            st.warning("⚠️ Medium confidence prediction. Consider checking the image quality.")
        else:
            st.error("❌ Low confidence prediction. The image might be unclear or not a digit.")
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.write("Please make sure you're uploading a valid image file (PNG or JPG).")
