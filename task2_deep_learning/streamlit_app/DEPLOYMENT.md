# Streamlit Cloud Deployment Guide

## Quick Deployment Steps

1. **Push to GitHub**: Ensure your code is pushed to GitHub repository
2. **Go to Streamlit Cloud**: Visit [share.streamlit.io](https://share.streamlit.io)
3. **Connect Repository**: Connect your GitHub repository
4. **Configure Deployment**:
   - **Main file path**: `task2_deep_learning/streamlit_app/app.py`
   - **Branch**: `main`
   - **Python version**: 3.9 (default)

## Files Included for Deployment

- `app.py` - Main Streamlit application
- `mnist_model.h5` - Trained TensorFlow model (2.7 MB)
- `requirements.txt` - Python dependencies with specific versions
- `.streamlit/config.toml` - Streamlit configuration
- `packages.txt` - System dependencies (empty for this app)

## Common Issues and Solutions

### Model Loading Issues
- The app tries multiple paths to find the model file
- Model is loaded with `compile=False` for better compatibility
- Check the sidebar for model loading status

### Memory Issues
- Model size is 2.7 MB (well within Streamlit Cloud limits)
- Uses `@st.cache_resource` for efficient model caching

### Dependencies
- All dependencies are pinned to specific versions for stability
- TensorFlow 2.15.0 is used for better compatibility

## Testing Locally

Before deploying, test locally:
```bash
cd task2_deep_learning/streamlit_app
python -m streamlit run app.py
```

## Deployment URL

Once deployed, your app will be available at:
`https://[your-app-name].streamlit.app`
