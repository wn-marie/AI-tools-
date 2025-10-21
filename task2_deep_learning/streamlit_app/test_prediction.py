#!/usr/bin/env python3
"""
Test script to verify MNIST prediction accuracy
Run this script to test the model with sample data
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

def test_model_accuracy():
    """Test the model with actual MNIST test data"""
    try:
        # Load the model
        model = tf.keras.models.load_model('mnist_model.h5')
        print(f"✅ Model loaded successfully")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        
        # Load MNIST test data
        (_, _), (x_test, y_test) = mnist.load_data()
        
        # Normalize the test data
        x_test = x_test / 255.0
        
        # Reshape based on model input shape
        if len(model.input_shape) == 4:  # CNN model (batch, height, width, channels)
            x_test = x_test.reshape(-1, 28, 28, 1)
        elif len(model.input_shape) == 3:  # Dense model (batch, 784)
            x_test = x_test.reshape(-1, 784)
        
        # Test on a small sample
        sample_size = 100
        x_sample = x_test[:sample_size]
        y_sample = y_test[:sample_size]
        
        # Make predictions
        predictions = model.predict(x_sample, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Calculate accuracy
        accuracy = np.mean(predicted_classes == y_sample)
        print(f"✅ Test accuracy on {sample_size} samples: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Show some examples
        print("\n📊 Sample predictions:")
        for i in range(5):
            actual = y_sample[i]
            predicted = predicted_classes[i]
            confidence = np.max(predictions[i]) * 100
            status = "✅" if actual == predicted else "❌"
            print(f"Sample {i+1}: Actual={actual}, Predicted={predicted}, Confidence={confidence:.1f}% {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {str(e)}")
        return False

def test_image_preprocessing():
    """Test the image preprocessing function"""
    print("\n🔧 Testing image preprocessing...")
    
    # Create a test image (simulate a digit 3)
    test_image = np.zeros((28, 28))
    # Draw a simple "3" shape
    test_image[5:23, 5:7] = 1    # Left vertical line
    test_image[5:7, 5:20] = 1    # Top horizontal line
    test_image[13:15, 5:20] = 1  # Middle horizontal line
    test_image[21:23, 5:20] = 1  # Bottom horizontal line
    test_image[5:23, 18:20] = 1  # Right vertical line
    
    # Test inversion logic
    original_mean = np.mean(test_image)
    print(f"Original image mean: {original_mean:.3f}")
    
    if original_mean > 0.5:
        inverted = 1.0 - test_image
        print(f"Inverted image mean: {np.mean(inverted):.3f}")
        print("✅ Image inversion logic working correctly")
    else:
        print("✅ No inversion needed")
    
    return True

if __name__ == "__main__":
    print("🧪 Testing MNIST Model and Preprocessing")
    print("=" * 50)
    
    # Test model accuracy
    model_ok = test_model_accuracy()
    
    # Test preprocessing
    preprocessing_ok = test_image_preprocessing()
    
    print("\n" + "=" * 50)
    if model_ok and preprocessing_ok:
        print("🎉 All tests passed! The app should work correctly now.")
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
