#!/usr/bin/env python3
"""
Script to check if the model file exists and is accessible
This helps debug model loading issues in Streamlit Cloud
"""

import os
import sys

def check_model_file():
    print("=== Model File Check ===")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {__file__}")
    print(f"Script directory: {os.path.dirname(__file__)}")
    
    # List all files in current directory
    print("\nFiles in current directory:")
    try:
        files = os.listdir('.')
        for file in files:
            print(f"  - {file}")
    except Exception as e:
        print(f"Error listing files: {e}")
    
    # Check for model file in various locations
    possible_paths = [
        'mnist_model.h5',
        './mnist_model.h5',
        os.path.join(os.path.dirname(__file__), 'mnist_model.h5'),
        os.path.join(os.getcwd(), 'mnist_model.h5')
    ]
    
    print("\nChecking for model file:")
    model_found = False
    for path in possible_paths:
        exists = os.path.exists(path)
        print(f"  {path}: {'EXISTS' if exists else 'NOT FOUND'}")
        if exists:
            try:
                size = os.path.getsize(path)
                print(f"    Size: {size:,} bytes ({size/1024/1024:.2f} MB)")
                model_found = True
            except Exception as e:
                print(f"    Error getting size: {e}")
    
    if not model_found:
        print("\n❌ Model file not found in any expected location!")
        return False
    else:
        print("\n✅ Model file found!")
        return True

if __name__ == "__main__":
    check_model_file()
