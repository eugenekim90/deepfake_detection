# 🔍 Deepfake Detection System

A comprehensive deepfake detection system with an interactive web dashboard built using deep learning and Streamlit. This project provides both model training capabilities and a user-friendly interface for real-time deepfake detection.

## 🌟 Features

### 🎯 Interactive Web Dashboard (`app.py`)
- **Real-time Detection**: Upload images and get instant deepfake predictions
- **Grad-CAM Visualization**: See exactly where the model focuses when making decisions
- **Smart Image Processing**: Handles multiple formats (PNG, JPG, JPEG, BMP, TIFF, WebP)
- **Robust Preprocessing**: Automatic orientation correction, format conversion, and resizing
- **Visual Results**: Clean, intuitive interface with confidence scores and probability meters
- **Model Explainability**: Gradient-weighted Class Activation Mapping for transparency

### 🧠 Model Capabilities
- **Binary Classification**: Distinguishes between real and fake images
- **CNN Architecture**: Optimized convolutional neural network
- **Transfer Learning**: Built on proven deep learning foundations
- **High Accuracy**: Trained on diverse datasets for robust performance

### 📊 Exploratory Data Analysis
- **Class Distribution Analysis**: Visualize dataset balance
- **Image Statistics**: RGB analysis, resolution distribution
- **Quality Metrics**: Sharpness, contrast, and luminance analysis
- **FFT Spectrum Analysis**: Frequency domain characteristics
- **Sample Visualization**: Grid displays of real vs fake examples

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/eugenekim90/deepfake_detection.git
cd deepfake_detection

# Install dependencies
pip install -r requirements.txt
```

### Running the Dashboard
```bash
# Launch the interactive web dashboard
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 🎮 Using the Dashboard

### 1. **Upload Image**
- Click "Browse files" or drag & drop an image
- Supports: PNG, JPG, JPEG, BMP, TIFF, WebP formats
- Maximum file size: 10MB
- Minimum resolution: 32x32 pixels

### 2. **View Results**
- **Prediction**: REAL or FAKE classification
- **Confidence Score**: Probability percentage (0-100%)
- **Progress Bar**: Visual representation of confidence
- **Image Info**: Displays dimensions, format, and aspect ratio

### 3. **Grad-CAM Analysis**
- **Attention Heatmap**: Shows model focus areas
- **Overlay Visualization**: Heatmap overlaid on original image
- **Color Coding**: 
  - 🔴 Red areas = High attention (important for decision)
  - 🔵 Blue areas = Low attention
- **Explainability**: Understand what features influenced the prediction

## 🏗️ Project Structure

```
deepfake_detection/
├── app.py                 # Main Streamlit dashboard
├── notebook.ipynb         # Training and analysis notebook
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .gitignore            # Git ignore rules
├── best_model.keras      # Trained model (local only)
├── eda/                  # Exploratory Data Analysis
│   ├── class_balance.png
│   ├── fft_spectrum.png
│   ├── image_stats.csv
│   ├── luminance_contrast.png
│   ├── resolution.png
│   ├── rgb_stats.png
│   └── sharpness.png
├── runs/                 # Training runs and checkpoints
│   └── exp01/
│       ├── ckpt_ft.keras
│       └── ckpt_head.keras
└── data/                 # Dataset (local only)
```

## 🔧 Technical Details

### Model Architecture
- **Input Size**: 160x160x3 RGB images
- **Architecture**: Convolutional Neural Network
- **Output**: Binary classification (Real: 0, Fake: 1)
- **Activation**: Sigmoid for probability output
- **Loss Function**: Binary cross-entropy

### Image Preprocessing Pipeline
1. **Format Validation**: Check file type and size
2. **Orientation Correction**: Fix EXIF rotation issues
3. **Color Mode Handling**: Convert RGBA, CMYK, etc. to RGB
4. **Smart Resizing**: Maintain aspect ratio with center cropping
5. **Normalization**: Scale pixel values to [0, 1] range

### Grad-CAM Implementation
- **Layer Selection**: Automatically finds last convolutional layer
- **Gradient Computation**: Backpropagation through target class
- **Heatmap Generation**: Weighted combination of feature maps
- **Visualization**: Jet colormap overlay on original image

## 📈 Performance Metrics

The model has been evaluated on:
- **Training Accuracy**: Monitored during training
- **Validation Performance**: Cross-validation results
- **Test Set Evaluation**: Final performance metrics
- **Grad-CAM Validation**: Visual inspection of attention maps

## 🎯 Use Cases

### 1. **Content Moderation**
- Social media platforms
- News verification
- Digital forensics

### 2. **Research & Education**
- Academic research
- Student projects
- AI explainability studies

### 3. **Personal Use**
- Verify suspicious images
- Educational demonstrations
- Understanding AI decision-making

## 🛠️ Development

### Training New Models
```bash
# Open the training notebook
jupyter notebook notebook.ipynb
```

### Model Files
- `best_model.keras`: Main production model
- `runs/exp01/ckpt_ft.keras`: Fine-tuned checkpoint
- `runs/exp01/ckpt_head.keras`: Classification head checkpoint

### Dependencies
- **TensorFlow**: Deep learning framework
- **Streamlit**: Web dashboard framework
- **PIL/Pillow**: Image processing
- **NumPy**: Numerical computations
- **Matplotlib**: Visualization
- **Pandas**: Data manipulation

## 🔒 Privacy & Security

- **Local Processing**: All analysis happens on your machine
- **No Data Storage**: Uploaded images are not saved
- **No External Calls**: Model runs entirely offline
- **Privacy First**: Your images never leave your device

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**⚡ Ready to detect deepfakes? Run `streamlit run app.py` and start analyzing!** 