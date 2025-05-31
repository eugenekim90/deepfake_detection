# Simple Deepfake Detection with CNN

This project implements a simple CNN-based deepfake detection system. The model is designed to be lightweight and easy to train while still providing reasonable accuracy.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure your dataset is organized as follows:
- `train.csv`, `valid.csv`, and `test.csv` files containing image paths and labels
- Images stored in the `real_vs_fake/real-vs-fake` directory

## Training the Model

To train the model, simply run:
```bash
python deepfake_detection.py
```

The script will:
1. Load and preprocess the images
2. Train a simple CNN model
3. Save the trained model as 'deepfake_detection_model.h5'

## Model Architecture

The model uses a simple CNN architecture with:
- 3 convolutional layers with max pooling
- Dense layers with dropout for classification
- Binary cross-entropy loss for binary classification (real vs fake)

## Parameters

You can modify the following parameters in the script:
- `BATCH_SIZE`: Number of samples per training batch (default: 32)
- `EPOCHS`: Number of training epochs (default: 10)
- `IMAGE_SIZE`: Input image dimensions (default: 128x128)
- `MAX_SAMPLES`: Maximum number of samples to use for training (default: 1000)

## Notes

- The model uses early stopping to prevent overfitting
- Images are normalized to [0, 1] range
- The model is saved in HDF5 format for easy loading and inference 