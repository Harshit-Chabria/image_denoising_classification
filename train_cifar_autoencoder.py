"""
train_cifar_autoencoder.py
--------------------------
Step 1 of the CIFAR-10 pipeline:
  - Load CIFAR-10 dataset (32x32 RGB images)
  - Add Gaussian noise to simulate real-world conditions
  - Build and train a Convolutional Autoencoder to remove noise
  - Save the trained model as 'cifar_autoencoder.keras'
  - Save denoised images as .npy files for the classifier to use
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# ──────────────────────────────────────────────
# 1. LOAD & PREPROCESS CIFAR-10 DATASET
# ──────────────────────────────────────────────
print("[INFO] Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

print(f"[INFO] Training samples : {x_train.shape[0]}")
print(f"[INFO] Testing  samples : {x_test.shape[0]}")
print(f"[INFO] Clean image shape: {x_train.shape[1:]}")

# ──────────────────────────────────────────────
# 2. ADD GAUSSIAN NOISE
# ──────────────────────────────────────────────
# CIFAR is more complex, so a noise factor of 0.2 is tough but reasonable
noise_factor = 0.2
print(f"[INFO] Adding Gaussian noise (noise_factor={noise_factor})...")

np.random.seed(42)   # reproducibility
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy  = x_test  + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

# Clip to keep valid pixel range [0, 1]
x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
x_test_noisy  = np.clip(x_test_noisy,  0.0, 1.0)

# ──────────────────────────────────────────────
# 3. BUILD CONVOLUTIONAL AUTOENCODER
# ──────────────────────────────────────────────
print("[INFO] Building CIFAR-10 Convolutional Autoencoder...")

input_img = layers.Input(shape=(32, 32, 3), name="encoder_input")

# ----- ENCODER -----
# Extract features and compress the image
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="enc_conv1")(input_img)
x = layers.MaxPooling2D((2, 2), padding="same", name="enc_pool1")(x)   # 32x32 → 16x16
x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="enc_conv2")(x)
encoded = layers.MaxPooling2D((2, 2), padding="same", name="enc_pool2")(x)  # 16x16 → 8x8

# ----- DECODER -----
# Reconstruct the image. BatchNorm is critical to prevent zero-collapse
x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="dec_conv1")(encoded)
x = layers.BatchNormalization(name="dec_bn1")(x)
x = layers.UpSampling2D((2, 2), name="dec_up1")(x)   # 8x8 → 16x16
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="dec_conv2")(x)
x = layers.BatchNormalization(name="dec_bn2")(x)
x = layers.UpSampling2D((2, 2), name="dec_up2")(x)   # 16x16 → 32x32
# Final layer: 3 filters (RGB) + sigmoid to output image in [0, 1]
decoded = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same", name="dec_output")(x)

# Combine encoder + decoder into the full autoencoder model
autoencoder = models.Model(input_img, decoded, name="cifar10_autoencoder")
autoencoder.summary()

# ──────────────────────────────────────────────
# 4. COMPILE & TRAIN
# ──────────────────────────────────────────────
# Training runs for 15 epochs; CIFAR-10 requires more epochs than MNIST to reconstruct properly.
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")

print("[INFO] Training CIFAR-10 Autoencoder (15 epochs)...")
autoencoder.fit(
    x_train_noisy,  # input:  noisy images
    x_train,        # target: clean images
    epochs=15,
    batch_size=128,
    shuffle=True,
    validation_data=(x_test_noisy, x_test),
    verbose=1
)

# ──────────────────────────────────────────────
# 4b. SANITY CHECK
# ──────────────────────────────────────────────
sample_out = autoencoder.predict(x_test_noisy[:100], verbose=0)
out_max = sample_out.max()
print(f"[CHECK] Sample output max pixel value: {out_max:.4f}  (should be > 0.5)")
if out_max < 0.1:
    raise RuntimeError("Autoencoder output is degenerate! Model collapsed to near zeros.")

# ──────────────────────────────────────────────
# 5. SAVE MODEL
# ──────────────────────────────────────────────
model_path = "cifar_autoencoder.keras"
autoencoder.save(model_path)
print(f"[INFO] Autoencoder model saved to '{model_path}'")

# ──────────────────────────────────────────────
# 6. GENERATE & SAVE DENOISED IMAGES
# ──────────────────────────────────────────────
print("[INFO] Generating denoised CIFAR-10 images from trained autoencoder...")

# Predict in batches to avoid OOM issues on standard machines
x_train_denoised = autoencoder.predict(x_train_noisy, batch_size=256, verbose=1)
x_test_denoised  = autoencoder.predict(x_test_noisy,  batch_size=256, verbose=1)

print(f"[CHECK] Denoised range: [{x_train_denoised.min():.4f}, {x_train_denoised.max():.4f}]")

np.save("cifar_x_train_denoised.npy", x_train_denoised)
np.save("cifar_x_test_denoised.npy",  x_test_denoised)
np.save("cifar_y_train.npy", y_train)
np.save("cifar_y_test.npy",  y_test)

print("[INFO] Saved: cifar_x_train_denoised.npy, cifar_x_test_denoised.npy, cifar_y_train.npy, cifar_y_test.npy")
print("[DONE] CIFAR-10 Autoencoder training complete!")
