# 🔍 Deepfake Detection System

Interactive web dashboard for real-time deepfake detection using CNN with Grad-CAM visualization.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
git clone https://github.com/eugenekim90/deepfake_detection.git
cd deepfake_detection
pip install -r requirements.txt
```

### 2. Run Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

## 📊 Dataset

**Source**: [Kaggle Real vs Fake Faces Dataset](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection)
- 1,081 real face images
- 960 fake face images  
- High-quality images for training

**Setup**:
```bash
# Download dataset from Kaggle
# Extract to: data/real_vs_fake/
# Structure:
# data/
# ├── train.csv
# ├── valid.csv  
# ├── test.csv
# └── real_vs_fake/
#     ├── real/
#     └── fake/
```

## 🎯 Training

### Step 1: Prepare Data
```python
# Open notebook.ipynb
jupyter notebook notebook.ipynb

# Run cells to:
# 1. Load and preprocess images
# 2. Create train/validation splits
# 3. Augment data
```

### Step 2: Train Model
```python
# In notebook, run training cells:
# - CNN architecture setup
# - Model compilation
# - Training with early stopping
# - Save best model as 'best_model.keras'
```

### Step 3: Evaluate
```python
# Generate performance metrics
# Create visualizations
# Test Grad-CAM functionality
```

## 🚀 Deployment

### Local Deployment
```bash
# Ensure model file exists
ls best_model.keras

# Run dashboard
streamlit run app.py
```

### Cloud Deployment (Streamlit Cloud)
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Deploy automatically

### Docker Deployment
```bash
# Create Dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]

# Build and run
docker build -t deepfake-detection .
docker run -p 8501:8501 deepfake-detection
```

## 🎮 Dashboard Features

- **Upload Image** → Get instant REAL/FAKE prediction
- **Confidence Score** → See probability percentage  
- **Grad-CAM** → Visualize where model focuses
- **Multiple Formats** → PNG, JPG, JPEG, BMP, TIFF, WebP

## 📁 Project Structure

```
├── app.py              # Streamlit dashboard
├── notebook.ipynb      # Training notebook
├── best_model.keras    # Trained model
├── requirements.txt    # Dependencies
├── eda/               # Analysis results
└── data/              # Dataset (local)
```

## 🔧 Model Details

- **Architecture**: CNN with 160x160 input
- **Output**: Binary classification (Real/Fake)
- **Visualization**: Grad-CAM attention maps
- **Performance**: Trained on 2K+ images

---

**Ready to detect deepfakes? Run `streamlit run app.py`** 🚀 