"""
train_autoencoder.py
--------------------
Step 1 of the pipeline:
  - Load MNIST dataset
  - Add Gaussian noise to simulate real-world conditions
  - Build and train a Convolutional Autoencoder to remove noise
  - Save the trained model as 'autoencoder.keras'
  - Save denoised images as .npy files for the classifier to use
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# ──────────────────────────────────────────────
# 1. LOAD & PREPROCESS MNIST DATASET
# ──────────────────────────────────────────────
print("[INFO] Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

# Reshape to (samples, 28, 28, 1) — required by Conv2D layers
x_train = x_train.reshape(-1, 28, 28, 1)
x_test  = x_test.reshape(-1, 28, 28, 1)

print(f"[INFO] Training samples : {x_train.shape[0]}")
print(f"[INFO] Testing  samples : {x_test.shape[0]}")
print(f"[INFO] Clean image pixel range: [{x_train.min():.3f}, {x_train.max():.3f}]")

# ──────────────────────────────────────────────
# 2. ADD GAUSSIAN NOISE
# ──────────────────────────────────────────────
noise_factor = 0.5   # controls how much noise is injected
print(f"[INFO] Adding Gaussian noise (noise_factor={noise_factor})...")

np.random.seed(42)   # reproducibility
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy  = x_test  + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

# Clip to keep valid pixel range [0, 1]
x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
x_test_noisy  = np.clip(x_test_noisy,  0.0, 1.0)
print(f"[INFO] Noisy image pixel range: [{x_train_noisy.min():.3f}, {x_train_noisy.max():.3f}]")

# ──────────────────────────────────────────────
# 3. BUILD CONVOLUTIONAL AUTOENCODER
# ──────────────────────────────────────────────
# BatchNormalization layers in the decoder prevent the degenerate case where
# the model minimizes MSE by outputting all zeros (which MNIST backgrounds are).
print("[INFO] Building Convolutional Autoencoder...")

input_img = layers.Input(shape=(28, 28, 1), name="encoder_input")

# ----- ENCODER -----
# Extract features and compress the image into a latent representation
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="enc_conv1")(input_img)
x = layers.MaxPooling2D((2, 2), padding="same", name="enc_pool1")(x)   # 28x28 → 14x14
x = layers.Conv2D(16, (3, 3), activation="relu", padding="same", name="enc_conv2")(x)
encoded = layers.MaxPooling2D((2, 2), padding="same", name="enc_pool2")(x)  # 14x14 → 7x7

# ----- DECODER -----
# Reconstruct the clean image from the compressed latent representation.
# BatchNorm after each Conv2D stabilises activations and prevents zero collapse.
x = layers.Conv2D(16, (3, 3), activation="relu", padding="same", name="dec_conv1")(encoded)
x = layers.BatchNormalization(name="dec_bn1")(x)
x = layers.UpSampling2D((2, 2), name="dec_up1")(x)   # 7x7 → 14x14
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="dec_conv2")(x)
x = layers.BatchNormalization(name="dec_bn2")(x)
x = layers.UpSampling2D((2, 2), name="dec_up2")(x)   # 14x14 → 28x28
# Final layer: 1 filter + sigmoid to output a grayscale image in [0, 1]
decoded = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same", name="dec_output")(x)

# Combine encoder + decoder into the full autoencoder model
autoencoder = models.Model(input_img, decoded, name="convolutional_autoencoder")
autoencoder.summary()

# ──────────────────────────────────────────────
# 4. COMPILE & TRAIN
# ──────────────────────────────────────────────
# Input  = noisy images   (what the model receives)
# Output = clean images   (what the model should reconstruct)
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")

print("[INFO] Training Autoencoder (10 epochs)...")
autoencoder.fit(
    x_train_noisy,  # input:  noisy images
    x_train,        # target: clean images
    epochs=10,
    batch_size=128,
    shuffle=True,
    validation_data=(x_test_noisy, x_test),
    verbose=1
)

# ──────────────────────────────────────────────
# 4b. SANITY CHECK — ensure outputs are not degenerate zeros
# ──────────────────────────────────────────────
sample_out = autoencoder.predict(x_test_noisy[:100], verbose=0)
out_max = sample_out.max()
print(f"[CHECK] Sample output max pixel value: {out_max:.4f}  (should be > 0.1)")
if out_max < 0.05:
    raise RuntimeError(
        f"Autoencoder output is degenerate (max={out_max:.5f})! "
        "The model is collapsing to zeros. Check architecture and data."
    )

# ──────────────────────────────────────────────
# 5. SAVE MODEL
# ──────────────────────────────────────────────
model_path = "autoencoder.keras"
autoencoder.save(model_path)
print(f"[INFO] Autoencoder model saved to '{model_path}'")

# ──────────────────────────────────────────────
# 6. GENERATE & SAVE DENOISED IMAGES
# ──────────────────────────────────────────────
# These denoised images will be the input to the CNN classifier
print("[INFO] Generating denoised images from trained autoencoder...")

x_train_denoised = autoencoder.predict(x_train_noisy, verbose=1)
x_test_denoised  = autoencoder.predict(x_test_noisy,  verbose=1)

print(f"[CHECK] Denoised range: [{x_train_denoised.min():.4f}, {x_train_denoised.max():.4f}]")

# Save as .npy files so train_classifier.py can load them directly
np.save("x_train_denoised.npy", x_train_denoised)
np.save("x_test_denoised.npy",  x_test_denoised)
np.save("y_train.npy", y_train)
np.save("y_test.npy",  y_test)

print("[INFO] Saved: x_train_denoised.npy, x_test_denoised.npy, y_train.npy, y_test.npy")
print("[DONE] Autoencoder training complete!")
