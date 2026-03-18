# DL Image Denoising & Classification System

An end-to-end deep learning pipeline for image denoising and classification using:
- **Convolutional Autoencoder** — removes Gaussian noise from images
- **CNN Classifier** — classifies denoised images (digits 0–9)
- **Flask Web App** — simple browser-based UI to run the full pipeline

---

## Project Structure

```
image_denoising_classification/
├── train_autoencoder.py      # Train & save the denoising autoencoder
├── train_classifier.py       # Train & save the CNN classifier
├── app.py                    # Flask web application
├── templates/
│   └── index.html            # Frontend HTML template
├── static/
│   └── images/               # Auto-created; stores input/output images
├── requirements.txt
└── README.md
```

---

## How to Run

### Prerequisites
- Python 3.8+
- pip

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train the Autoencoder
This will load MNIST, add noise, train the autoencoder, and save:
- `autoencoder.h5`
- `x_train_denoised.npy`, `x_test_denoised.npy`
- `y_train.npy`, `y_test.npy`

```bash
python train_autoencoder.py
```
> ⏱ Expected time: ~3–5 minutes

### Step 3 — Train the CNN Classifier
This loads the denoised images and trains the digit classifier, saving:
- `classifier.h5`

```bash
python train_classifier.py
```
> ⏱ Expected time: ~2–3 minutes

### Step 4 — Start the Flask Server
```bash
python app.py
```
Then open your browser at: **http://127.0.0.1:5000**

---

## Using the Web App

1. Open `http://127.0.0.1:5000`
2. Upload any grayscale image of a handwritten digit (PNG/JPG)
   - You can draw one in Microsoft Paint and save it
   - Or use any digit image from the internet
3. Click **Run Pipeline**
4. The app will display:
   - 📷 Original (preprocessed 28×28 grayscale)
   - 🌫️ Noisy version (Gaussian noise added)
   - ✨ Denoised output (autoencoder reconstruction)
   - 🔢 Predicted digit class + confidence %

---

## Pipeline Overview

```
User Image
    │
    ▼
Preprocess (grayscale → 28×28 → normalize)
    │
    ▼
Add Gaussian Noise (noise_factor=0.5)
    │
    ▼
Convolutional Autoencoder  ──→  Denoised Image
    │
    ▼
CNN Classifier  ──→  Predicted Digit (0–9)
    │
    ▼
Display in Browser
```

---

## Model Architectures

### Autoencoder
| Layer           | Type          | Output Shape |
|-----------------|---------------|--------------|
| enc_conv1       | Conv2D(32)    | 28×28×32     |
| enc_pool1       | MaxPooling2D  | 14×14×32     |
| enc_conv2       | Conv2D(16)    | 14×14×16     |
| enc_pool2       | MaxPooling2D  | 7×7×16       |
| dec_conv1       | Conv2D(16)    | 7×7×16       |
| dec_up1         | UpSampling2D  | 14×14×16     |
| dec_conv2       | Conv2D(32)    | 14×14×32     |
| dec_up2         | UpSampling2D  | 28×28×32     |
| dec_output      | Conv2D(1, σ)  | 28×28×1      |

### CNN Classifier
| Layer    | Type          | Output Shape |
|----------|---------------|--------------|
| conv1    | Conv2D(32)    | 28×28×32     |
| pool1    | MaxPooling2D  | 14×14×32     |
| conv2    | Conv2D(64)    | 14×14×64     |
| pool2    | MaxPooling2D  | 7×7×64       |
| flatten  | Flatten       | 3136         |
| fc1      | Dense(128)    | 128          |
| output   | Dense(10, SM) | 10           |
