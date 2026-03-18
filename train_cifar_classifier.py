"""
train_cifar_classifier.py
-------------------------
Step 2 of the CIFAR-10 pipeline:
  - Load denoised RGB images produced by the autoencoder (from .npy files)
  - Build a deeper CNN for 10-class object classification
  - Train and evaluate the model
  - Save the trained classifier as 'cifar_classifier.keras'
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import os

# ──────────────────────────────────────────────
# 1. LOAD DENOISED IMAGES & LABELS
# ──────────────────────────────────────────────
print("[INFO] Loading denoised CIFAR-10 images from .npy files...")

if not os.path.exists("cifar_x_train_denoised.npy"):
    raise FileNotFoundError(
        "Denoised training images not found! "
        "Please run 'python train_cifar_autoencoder.py' first."
    )

x_train = np.load("cifar_x_train_denoised.npy")
x_test  = np.load("cifar_x_test_denoised.npy")
y_train = np.load("cifar_y_train.npy")
y_test  = np.load("cifar_y_test.npy")

print(f"[INFO] Training samples : {x_train.shape[0]}")
print(f"[INFO] Testing  samples : {x_test.shape[0]}")

# ──────────────────────────────────────────────
# 2. ONE-HOT ENCODE LABELS
# ──────────────────────────────────────────────
num_classes = 10  # airplane, auto, bird, cat, deer, dog, frog, horse, ship, truck
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat  = to_categorical(y_test,  num_classes)

# ──────────────────────────────────────────────
# 3. BUILD CNN CLASSIFIER
# ──────────────────────────────────────────────
print("[INFO] Building CIFAR-10 CNN Classifier...")

model = models.Sequential(name="cifar10_classifier_cnn")

# --- Deeper network needed for CIFAR-10 over MNIST ---
model.add(layers.Input(shape=(32, 32, 3)))
model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2), name="pool1"))   # 32x32 → 16x16

model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2), name="pool2"))   # 16x16 → 8x8

model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv3"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2), name="pool3"))   # 8x8 → 4x4

model.add(layers.Flatten(name="flatten"))
model.add(layers.Dense(128, activation="relu", name="fc1"))
model.add(layers.Dropout(0.5, name="dropout1"))

model.add(layers.Dense(num_classes, activation="softmax", name="output"))

model.summary()

# ──────────────────────────────────────────────
# 4. COMPILE & TRAIN
# ──────────────────────────────────────────────
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("[INFO] Training CIFAR-10 Classifier (15 epochs)...")
model.fit(
    x_train, y_train_cat,
    epochs=15,
    batch_size=128,
    shuffle=True,
    validation_data=(x_test, y_test_cat),
    verbose=1
)

# ──────────────────────────────────────────────
# 5. EVALUATE ON TEST SET
# ──────────────────────────────────────────────
print("[INFO] Evaluating CNN on test set...")
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"[RESULT] CIFAR-10 Test Accuracy : {test_acc * 100:.2f}%")
print(f"[RESULT] CIFAR-10 Test Loss     : {test_loss:.4f}")

# ──────────────────────────────────────────────
# 6. SAVE MODEL
# ──────────────────────────────────────────────
model_path = "cifar_classifier.keras"
model.save(model_path)
print(f"[INFO] Classifier model saved to '{model_path}'")
print("[DONE] CIFAR-10 CNN training complete!")
