# 🔍 Deepfake Detection System

A lightweight CNN-based deepfake detection system with interactive dashboard. **Train your own model** from scratch using the provided notebook and dataset.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
git clone https://github.com/eugenekim90/deepfake_detection.git
cd deepfake_detection
pip install -r requirements.txt
```

### 2. Get Dataset & Train Model
**You must train your own model first** - no pre-trained model is provided.

## 📊 Dataset

**Source**: [Kaggle 140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces/code?datasetId=501529&sortBy=commentCount)
- **Real images**: 70k real faces (from Flickr)
- **Fake images**: 70k fake faces (GAN-generated)

**Setup**:
```bash
# 1. Download from Kaggle
# 2. Extract to: data/real_vs_fake/
```

## 🎯 Training (Required)

### Step 1: EDA & Data Analysis
```bash
# Open training notebook
jupyter notebook notebook.ipynb

# Run EDA cells to analyze:
# - Image sharpness distribution
# - Edge definition (real vs fake)
# - High-frequency details
# - Texture analysis
```

### Step 2: Train Your Model
```python
# In notebook, train the lightweight CNN:
# - 3 convolutional-pooling blocks
# - Dropout for regularization  
# - Data augmentation (flips, rotations)
# - Early stopping to prevent overfitting
# - Saves model as 'best_model.keras'
```

### Step 3: Model Architecture
- **Input**: 160x160x3 RGB images
- **Architecture**: Lightweight CNN (3 conv blocks)
- **Output**: Binary classification (Real=0, Fake=1)
- **Training**: ~30 epochs with early stopping
- **Augmentation**: Flips and rotations only

## 🚀 Deploy Dashboard

### After Training Your Model:
```bash
# Ensure your trained model exists
ls best_model.keras

# Run interactive dashboard
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### Cloud Deployment
1. Train model locally first
2. Upload trained model to your deployment
3. Deploy to Streamlit Cloud/Heroku/etc.

## 🎮 Dashboard Features

- **Upload Image** → Get REAL/FAKE prediction
- **Confidence Score** → Probability percentage
- **Grad-CAM Visualization** → See where model focuses
- **Multiple Formats** → PNG, JPG, JPEG, BMP, TIFF, WebP
- **Real-time Processing** → Instant results

## 🔬 Why This Matters

**Problem**: Anyone can generate realistic fake faces using free tools
- Criminals create fake IDs and profiles
- Fraudsters use fake photos for scams  
- Manual screening is impossible at scale

**Solution**: Automated detection using lightweight CNN
- Real faces have sharper edges and fine texture
- Fake faces lack high-frequency details
- Model learns these subtle differences

## 📁 Project Structure

```
├── app.py              # Streamlit dashboard
├── notebook.ipynb      # Training notebook (START HERE)
├── requirements.txt    # Dependencies
├── eda/               # Analysis results
├── data/              # Dataset (download separately)
└── best_model.keras   # Your trained model (after training)
```

**⚠️ Important: You must train the model first using `notebook.ipynb` before running the dashboard!** 