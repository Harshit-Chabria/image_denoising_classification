"""
app.py
------
Flask web application for the denoising + classification pipeline.

Routes:
  GET  /          → Homepage with upload form
  POST /predict   → Accepts image, runs pipeline, returns results

Pipeline per request:
  1. Load uploaded image → grayscale → resize 28x28 → normalize
  2. Add Gaussian noise (simulate real-world input)
  3. Denoise with autoencoder
  4. Classify with CNN
  5. Render results page with original, noisy, denoised images + predicted label
"""

import os
import uuid
import numpy as np
from flask import Flask, request, render_template
from PIL import Image, ImageOps
import tensorflow as tf

# ──────────────────────────────────────────────
# FLASK APP SETUP
# ──────────────────────────────────────────────
app = Flask(__name__)

# Folder to store all uploaded and processed images
STATIC_FOLDER = os.path.join("static", "images")
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Limit TensorFlow CPU threading to reduce RAM footprint on 512MB free tier
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# ──────────────────────────────────────────────
# LAZY-LOAD TRAINED MODELS (Saves Memory)
# ──────────────────────────────────────────────
# We don't load all 4 models at startup anymore to prevent OOM crashes.
# We hold them in a dictionary only when needed.
MODEL_CACHE = {}

def get_model(model_name, filename):
    if model_name not in MODEL_CACHE:
        print(f"[INFO] Lazy-loading {filename} into memory...")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"{filename} not found!")
        MODEL_CACHE[model_name] = tf.keras.models.load_model(filename)
    return MODEL_CACHE[model_name]

# CIFAR-10 Class labels mapping
CIFAR_LABELS = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

# ──────────────────────────────────────────────
# HELPER: Save a (28, 28, 1) numpy array as PNG
# ──────────────────────────────────────────────
def save_image_array(array, filename):
    """Convert a normalized float32 array to a PNG file (handles grayscale and RGB)."""
    # Scale [0,1] → [0,255] and convert to uint8
    img_array = (array * 255).astype(np.uint8)
    
    # Check shape to determine mode: (H, W, 1) -> Grayscale, (H, W, 3) -> RGB
    if img_array.shape[-1] == 1:
        img_array = img_array.squeeze()
        img = Image.fromarray(img_array, mode="L")
    else:
        img = Image.fromarray(img_array, mode="RGB")
        
    # Scale up for better visibility in the browser
    img = img.resize((150, 150), resample=Image.NEAREST)
    img.save(filename)


# ──────────────────────────────────────────────
# ROUTE: Homepage
# ──────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    """Render the homepage with the upload form."""
    return render_template("index.html")


# ──────────────────────────────────────────────
# ROUTE: Prediction Pipeline
# ──────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    Full inference pipeline:
      upload → preprocess → add noise → denoise → classify → display
    """
    # Get which dataset pipeline to run (mnist or cifar)
    dataset = request.form.get("dataset", "mnist")
    
    # --- 1. Validate file upload ---
    if "image" not in request.files or request.files["image"].filename == "":
        return render_template("index.html", error="Please select an image file to upload.", active_tab=dataset)

    file = request.files["image"]
    uid = uuid.uuid4().hex[:8]

    # --- 2. Preprocess uploaded image based on selected dataset ---
    try:
        img = Image.open(file)
        
        if dataset == "mnist":
            img = img.convert("L")   # Convert to grayscale
            # Invert black-on-white drawings to white-on-black (MNIST format)
            img = ImageOps.invert(img)
            img = img.resize((28, 28), resample=Image.LANCZOS)
            img_array = np.array(img).astype("float32") / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)
        else: # cifar
            img = img.convert("RGB") # Ensure RGB
            img = img.resize((32, 32), resample=Image.LANCZOS)
            img_array = np.array(img).astype("float32") / 255.0
            img_array = img_array.reshape(1, 32, 32, 3)
            
    except Exception as e:
        return render_template("index.html", error=f"Invalid image file: {str(e)}", active_tab=dataset)

    # Save original image
    original_path = os.path.join(STATIC_FOLDER, f"original_{uid}.png")
    save_image_array(img_array[0], original_path)

    # --- 3. Add Gaussian noise ---
    noise_factor = 0.5 if dataset == "mnist" else 0.2
    noisy_array = img_array + noise_factor * np.random.normal(size=img_array.shape)
    noisy_array = np.clip(noisy_array, 0.0, 1.0)
    
    noisy_path = os.path.join(STATIC_FOLDER, f"noisy_{uid}.png")
    save_image_array(noisy_array[0], noisy_path)

    # --- 4. Denoise with Autoencoder ---
    print(f"[INFO] Running {dataset} autoencoder denoising...")
    if dataset == "mnist":
        ae_model = get_model("autoencoder_mnist", "autoencoder.keras")
    else:
        ae_model = get_model("autoencoder_cifar", "cifar_autoencoder.keras")
        
    denoised_array = ae_model.predict(noisy_array, verbose=0)

    denoised_path = os.path.join(STATIC_FOLDER, f"denoised_{uid}.png")
    save_image_array(denoised_array[0], denoised_path)

    # --- 5. Classify with CNN ---
    print(f"[INFO] Running {dataset} CNN classifier...")
    if dataset == "mnist":
        clf_model = get_model("classifier_mnist", "classifier.keras")
    else:
        clf_model = get_model("classifier_cifar", "cifar_classifier.keras")
        
    predictions = clf_model.predict(denoised_array, verbose=0)
    
    pred_idx = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0])) * 100
    
    if dataset == "mnist":
        predicted_class = str(pred_idx)
        class_label = f"Class {predicted_class}"
    else:
        predicted_class = CIFAR_LABELS[pred_idx]
        class_label = predicted_class

    print(f"[RESULT] {dataset.upper()} prediction: {class_label} (confidence: {confidence:.1f}%)")

    # --- 6. Render results page ---
    return render_template(
        "index.html",
        active_tab=dataset,
        original_img=original_path.replace("\\", "/"),
        noisy_img=noisy_path.replace("\\", "/"),
        denoised_img=denoised_path.replace("\\", "/"),
        predicted_class=predicted_class,
        class_label=class_label,
        confidence=f"{confidence:.1f}"
    )


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("[INFO] Starting Flask server at http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
