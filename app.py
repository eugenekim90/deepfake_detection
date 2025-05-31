import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps, ExifTags
import matplotlib.pyplot as plt
import io

# Set page config
st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for compact UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
        font-size: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        font-size: 1.1rem;
    }
    .real-prediction {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
    }
    .fake-prediction {
        background: linear-gradient(135deg, #f44336 0%, #da190b 100%);
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .upload-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 2px dashed #dee2e6;
    }
    .metric-small {
        text-align: center;
        padding: 0.5rem;
        background: white;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin: 0.25rem;
    }
    .info-box {
        background: #e3f2fd;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the deepfake detection model"""
    from pathlib import Path
    model_path = Path("best_fake_real.keras")
    if not model_path.exists():
        st.error("❌ Model file not found")
        st.stop()
    
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

def fix_image_orientation(image):
    """Fix image orientation based on EXIF data"""
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        
        exif = image._getexif()
        if exif is not None:
            orientation_value = exif.get(orientation)
            if orientation_value == 3:
                image = image.rotate(180, expand=True)
            elif orientation_value == 6:
                image = image.rotate(270, expand=True)
            elif orientation_value == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, TypeError):
        pass
    return image

def validate_and_clean_image(uploaded_file):
    """Validate and clean uploaded image with enhanced format support"""
    try:
        # Check file size (max 10MB)
        if uploaded_file.size > 10 * 1024 * 1024:
            st.error("❌ File too large. Please upload an image smaller than 10MB.")
            return None
        
        # Try to open image
        image = Image.open(uploaded_file)
        
        # Fix orientation
        image = fix_image_orientation(image)
        
        # Handle different color modes more robustly
        original_mode = image.mode
        
        if image.mode == 'RGBA':
            # Handle transparency with white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background
        elif image.mode == 'LA':
            # Grayscale with alpha
            background = Image.new('RGB', image.size, (255, 255, 255))
            gray = image.convert('L')
            rgb_image = Image.merge('RGB', (gray, gray, gray))
            if len(image.split()) > 1:
                background.paste(rgb_image, mask=image.split()[-1])
            else:
                background = rgb_image
            image = background
        elif image.mode == 'P':
            # Palette mode - convert carefully
            if 'transparency' in image.info:
                image = image.convert('RGBA')
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            else:
                image = image.convert('RGB')
        elif image.mode == 'L':
            # Grayscale - convert to RGB
            image = Image.merge('RGB', (image, image, image))
        elif image.mode == 'CMYK':
            # CMYK to RGB conversion
            image = image.convert('RGB')
        elif image.mode == '1':
            # 1-bit pixels - convert to RGB
            image = image.convert('RGB')
        elif image.mode not in ['RGB']:
            # Any other mode - force convert to RGB
            image = image.convert('RGB')
        
        # Validate image dimensions
        width, height = image.size
        if width < 32 or height < 32:
            st.error("❌ Image too small. Minimum size is 32x32 pixels.")
            return None
        
        if width > 4096 or height > 4096:
            st.warning("⚠️ Large image detected. Resizing for processing...")
            # Resize very large images while maintaining aspect ratio
            image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        
        # Final validation - ensure RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
        
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        return None

def smart_crop_and_resize(image, target_size=(160, 160)):
    """Enhanced smart crop and resize with better aspect ratio handling"""
    try:
        # Get original dimensions
        width, height = image.size
        target_width, target_height = target_size
        
        # Calculate aspect ratios
        original_ratio = width / height
        target_ratio = target_width / target_height
        
        # Smart cropping strategy
        if abs(original_ratio - target_ratio) < 0.1:
            # Aspect ratios are close - just resize
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        elif original_ratio > target_ratio:
            # Image is wider - crop width (center crop)
            new_width = int(height * target_ratio)
            left = (width - new_width) // 2
            image = image.crop((left, 0, left + new_width, height))
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        else:
            # Image is taller - crop height (center crop)
            new_height = int(width / target_ratio)
            top = (height - new_height) // 2
            image = image.crop((0, top, width, top + new_height))
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        return image
        
    except Exception as e:
        st.error(f"Error in image processing: {str(e)}")
        return None

def normalize_image_array(img_array):
    """Normalize image array for consistent model input"""
    try:
        # Ensure float32 type
        img_array = img_array.astype('float32')
        
        # Check current range
        min_val = np.min(img_array)
        max_val = np.max(img_array)
        
        # Normalize based on current range
        if max_val <= 1.0:
            # Already in [0,1] range - scale to [0,255]
            img_array = img_array * 255.0
        elif max_val <= 255.0:
            # Already in [0,255] range - keep as is
            pass
        else:
            # Unknown range - normalize to [0,255]
            if max_val > min_val:
                img_array = ((img_array - min_val) / (max_val - min_val)) * 255.0
            else:
                img_array = np.zeros_like(img_array)
        
        # Ensure values are in valid range
        img_array = np.clip(img_array, 0.0, 255.0)
        
        return img_array
        
    except Exception as e:
        st.error(f"❌ Normalization error: {str(e)}")
        return None

def preprocess_image(image, target_size=(160, 160)):
    """Enhanced image preprocessing pipeline with robust normalization"""
    try:
        # Validate input
        if image is None:
            st.error("❌ No image provided for preprocessing")
            return None
        
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Smart crop and resize
        processed_image = smart_crop_and_resize(image, target_size)
        if processed_image is None:
            st.error("❌ Failed to resize image")
            return None
        
        # Convert to numpy array
        img_array = np.array(processed_image)
        
        # Validate array shape
        if len(img_array.shape) != 3:
            st.error(f"❌ Invalid image dimensions: {img_array.shape}")
            return None
        
        if img_array.shape[2] != 3:
            st.error(f"❌ Invalid number of channels: {img_array.shape[2]}, expected 3")
            return None
        
        # Normalize pixel values
        img_array = normalize_image_array(img_array)
        if img_array is None:
            return None
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Final validation
        expected_shape = (1, target_size[0], target_size[1], 3)
        if img_array.shape != expected_shape:
            st.error(f"❌ Final shape mismatch: {img_array.shape}, expected: {expected_shape}")
            return None
        
        # Check for invalid values
        if np.any(np.isnan(img_array)) or np.any(np.isinf(img_array)):
            st.error("❌ Invalid pixel values detected (NaN or Inf)")
            return None
        
        # Ensure values are in expected range
        if np.min(img_array) < 0 or np.max(img_array) > 255:
            st.warning(f"⚠️ Pixel values outside expected range: [{np.min(img_array):.2f}, {np.max(img_array):.2f}]")
            img_array = np.clip(img_array, 0.0, 255.0)
        
        return img_array
        
    except Exception as e:
        st.error(f"❌ Preprocessing error: {str(e)}")
        return None

def get_prediction(model, img_array):
    """Get prediction from model with error handling"""
    try:
        if img_array is None:
            return None
        
        # Validate input shape
        expected_shape = (1, 160, 160, 3)
        if img_array.shape != expected_shape:
            st.error(f"❌ Invalid input shape: {img_array.shape}, expected: {expected_shape}")
            return None
        
        # Make prediction
        prediction = model.predict(img_array, verbose=0)
        
        # Handle different output formats
        if isinstance(prediction, list):
            prediction = prediction[0]
        
        # Extract probability
        if prediction.shape[-1] == 1:
            # Single output (sigmoid)
            prob_real = float(prediction[0][0])
        elif prediction.shape[-1] == 2:
            # Two outputs (softmax)
            prob_real = float(prediction[0][1])
        else:
            # Multiple outputs - take first
            prob_real = float(prediction[0][0])
        
        # Validate probability range
        prob_real = np.clip(prob_real, 0.0, 1.0)
        
        return prob_real
        
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return None

def find_last_conv_layer(model):
    """Find the last convolutional layer"""
    try:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
        return None
    except Exception:
        return None

def generate_gradcam(model, img_array, layer_name):
    """Generate Grad-CAM heatmap with robust error handling"""
    try:
        if img_array is None or layer_name is None:
            return None
        
        # Create gradient model
        grad_model = tf.keras.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(layer_name).output, model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            
            # Handle different prediction formats
            if isinstance(predictions, list):
                predictions = predictions[0]
            
            if predictions.shape[-1] == 1:
                loss = predictions[0][0]
            elif predictions.shape[-1] == 2:
                loss = predictions[0][1]
            else:
                loss = predictions[0][0]
        
        # Generate heatmap
        output = conv_outputs[0]
        grads = tape.gradient(loss, conv_outputs)[0]
        
        # Pool gradients and create heatmap
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, output), axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        
        # Normalize
        heatmap_max = tf.reduce_max(heatmap)
        if heatmap_max > 0:
            heatmap = heatmap / heatmap_max
        
        # Resize to target size
        heatmap_expanded = tf.expand_dims(heatmap, -1)
        heatmap_resized = tf.image.resize(heatmap_expanded, [160, 160])
        heatmap_final = tf.squeeze(heatmap_resized).numpy()
        
        # Validate output
        if np.any(np.isnan(heatmap_final)) or np.any(np.isinf(heatmap_final)):
            return None
        
        return heatmap_final
        
    except Exception as e:
        return None

def create_overlay(image, heatmap, alpha=0.4):
    """Create overlay of heatmap on original image"""
    try:
        if heatmap is None:
            return np.array(image.resize((160, 160)))
        
        # Normalize heatmap
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        # Create colored heatmap
        heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Resize image to match heatmap
        image_resized = np.array(image.resize((160, 160))).astype(np.uint8)
        
        # Create overlay
        overlay = (heatmap_colored * alpha + image_resized * (1 - alpha)).astype(np.uint8)
        
        return overlay
        
    except Exception as e:
        return np.array(image.resize((160, 160)))

def display_image_info(image):
    """Display image information"""
    width, height = image.size
    mode = image.mode
    
    st.markdown(f"""
    <div class="info-box">
        📏 <strong>Size:</strong> {width} × {height} pixels<br>
        🎨 <strong>Mode:</strong> {mode}<br>
        📊 <strong>Aspect Ratio:</strong> {width/height:.2f}
    </div>
    """, unsafe_allow_html=True)

def main():
    # Compact header
    st.markdown('<h1 class="main-header">🔍 Deepfake Detection</h1>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    # Create two columns layout
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("**📁 Upload Image**")
        
        uploaded_file = st.file_uploader(
            "Choose image",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            # Validate and clean image
            image = validate_and_clean_image(uploaded_file)
            
            if image is not None:
                st.image(image, width=250)
                display_image_info(image)
            else:
                st.error("❌ Could not process the uploaded image")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if uploaded_file is not None and image is not None:
            st.markdown("**📊 Results**")
            
            # Process image with robust pipeline
            with st.spinner("🔄 Processing image..."):
                img_array = preprocess_image(image)
                
            if img_array is not None:
                # Get prediction
                with st.spinner("🤖 Analyzing with AI model..."):
                    prob_real = get_prediction(model, img_array)
                
                if prob_real is not None:
                    is_real = prob_real >= 0.5
                    result_text = "REAL" if is_real else "FAKE"
                    prediction_class = "real-prediction" if is_real else "fake-prediction"
                    
                    st.markdown(f"""
                    <div class="prediction-box {prediction_class}">
                        <strong>{result_text}</strong><br>
                        Probability: {prob_real:.1%}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Compact metrics
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f'<div class="metric-small"><small>Real</small><br><strong>{prob_real:.2%}</strong></div>', unsafe_allow_html=True)
                    with col_b:
                        st.markdown(f'<div class="metric-small"><small>Fake</small><br><strong>{1-prob_real:.2%}</strong></div>', unsafe_allow_html=True)
                    
                    st.progress(prob_real)
                    st.caption("← FAKE | REAL →")
                    
                    # Grad-CAM Visualization
                    st.markdown("**🔥 Grad-CAM Visualization**")
                    st.markdown("*Gradient-weighted Class Activation Mapping*")
                    
                    with st.spinner("🎯 Generating attention map..."):
                        last_conv = find_last_conv_layer(model)
                        
                        if last_conv:
                            heatmap = generate_gradcam(model, img_array, last_conv)
                            
                            if heatmap is not None:
                                # Create Grad-CAM visualization
                                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                                
                                # Heatmap
                                im = axes[0].imshow(heatmap, cmap='jet')
                                axes[0].set_title("Attention Heatmap", fontsize=12)
                                axes[0].axis('off')
                                
                                # Overlay
                                overlay = create_overlay(image, heatmap)
                                axes[1].imshow(overlay.astype(np.uint8))
                                axes[1].set_title("Overlay", fontsize=12)
                                axes[1].axis('off')
                                
                                plt.tight_layout()
                                plt.subplots_adjust(wspace=0.1)
                                st.pyplot(fig, use_container_width=True)
                                
                                st.caption("🔴 **Red areas** show where the model focused to make this prediction")
                            else:
                                st.warning("⚠️ Could not generate Grad-CAM for this image")
                        else:
                            st.warning("⚠️ No convolutional layers found for Grad-CAM")
                    
                    # Method info
                    with st.expander("ℹ️ About Grad-CAM"):
                        st.markdown("""
                        **Grad-CAM (Gradient-weighted Class Activation Mapping):**
                        - ⚡ Fast and efficient visualization technique
                        - 🎯 Shows where the model pays attention when making predictions
                        - 🔴 Red/hot areas = High attention regions that influenced the decision
                        - 🔵 Blue/cold areas = Low attention regions
                        - ✅ Helps understand what features the model considers important
                        - 🧠 Based on gradients flowing back through the last convolutional layer
                        """)
                else:
                    st.error("❌ Could not analyze the image")
            else:
                st.error("❌ Image preprocessing failed")
        else:
            st.info("👈 Upload an image to start analysis")

if __name__ == "__main__":
    main() 