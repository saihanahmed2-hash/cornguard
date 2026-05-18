

import os
import sys
import json

import logging
import numpy as np
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import tensorflow as tf
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# GPU memory growth
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import (
    Input, Dense, GlobalAveragePooling2D, Dropout,
    BatchNormalization,
)
from tensorflow.keras.models import Model
from flask import Flask, request, jsonify, send_from_directory

# ── Configuration ──
IMG_SIZE = (256, 256)
FINE_TUNE_AT = 150
CLASS_NAMES = ["common_rust", "gray_leaf_spot", "healthy", "northern_leaf_blight"]
NUM_CLASSES = len(CLASS_NAMES)
from huggingface_hub import hf_hub_download
from pathlib import Path

WEIGHTS_PATH = Path(hf_hub_download(
    repo_id="saihans/Multi_Head_Cnn",  # your HF repo
    filename="temp_ultimate_challenger.h5"               # exact filename you uploaded
))
# Paths — resolve relative to this script's location
# SCRIPT_DIR = Path(__file__).resolve().parent
# PROJECT_DIR = Path(r"c:\Users\saiha\Desktop\maize2")
# MODELS_DIR = PROJECT_DIR / "models"
# WEIGHTS_PATH = MODELS_DIR / "temp_ultimate_challenger.h5"

# Friendly metadata for each class
CLASS_INFO = {
    "common_rust": {
        "display": "Common Rust",
        "emoji": "🟠",
        "color": "#FF6B35",
        "description": "Caused by Puccinia sorghi. Small, circular to elongate pustules on both leaf surfaces. Pustules are cinnamon-brown to dark brown.",
        "treatment": "Apply fungicides like azoxystrobin or pyraclostrobin at first signs. Use resistant hybrids. Remove crop debris after harvest.",
    },
    "gray_leaf_spot": {
        "display": "Gray Leaf Spot",
        "emoji": "🔘",
        "color": "#78909C",
        "description": "Caused by Cercospora zeae-maydis. Rectangular, tan to gray lesions that run parallel to leaf veins. Can severely reduce yield.",
        "treatment": "Rotate crops (avoid corn-on-corn). Apply foliar fungicides at VT/R1. Use tolerant hybrids and reduce plant density.",
    },
    "healthy": {
        "display": "Healthy",
        "emoji": "🟢",
        "color": "#43A047",
        "description": "No disease detected. The leaf appears healthy with normal green coloration and no visible lesions or abnormalities.",
        "treatment": "No treatment needed. Continue regular crop management practices including proper irrigation, fertilization, and pest monitoring.",
    },
    "northern_leaf_blight": {
        "display": "Northern Leaf Blight",
        "emoji": "🟡",
        "color": "#FFA726",
        "description": "Caused by Exserohilum turcicum. Long, elliptical, grayish-green to tan lesions (1-6 inches). Can cause significant yield loss.",
        "treatment": "Apply fungicides at early disease onset. Plant resistant hybrids (Ht genes). Practice crop rotation and tillage to reduce inoculum.",
    },
}


# ── Losses (needed for compile, values don't matter for inference) ──
def underestimation_penalty_loss(y_true, y_pred):
    error = y_true - y_pred
    return tf.reduce_mean(tf.where(error > 0, 2.0 * tf.square(error), tf.square(error)))


def make_weighted_sparse_ce(class_weights_list):
    weights_as_floats = [float(w) for w in class_weights_list]
    def weighted_sparse_categorical_crossentropy(y_true, y_pred):
        weights_tensor = tf.constant(weights_as_floats, dtype=tf.float32)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        sample_weights = tf.gather(weights_tensor, tf.cast(y_true, tf.int32))
        return tf.reduce_mean(loss * sample_weights)
    weighted_sparse_categorical_crossentropy.__name__ = "weighted_sparse_ce"
    return weighted_sparse_categorical_crossentropy


# ── Build model (must match train_ultimate.py) ──
def build_model():
    inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="input_image")
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_tensor=inputs)
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    x = GlobalAveragePooling2D(name="gap")(base_model.output)
    x = BatchNormalization(name="gap_bn")(x)
    x = Dropout(0.4, name="gap_dropout")(x)

    shared = Dense(256, activation="relu", name="shared_dense")(x)
    shared = BatchNormalization(name="shared_bn")(shared)
    shared = Dropout(0.3, name="shared_dropout")(shared)

    cls = Dense(128, activation="relu", name="cls_dense1")(shared)
    cls = BatchNormalization(name="cls_bn")(cls)
    cls = Dropout(0.2, name="cls_dropout")(cls)
    class_output = Dense(NUM_CLASSES, activation="softmax", name="class_output")(cls)

    sev = Dense(128, activation="relu", name="sev_dense1")(shared)
    sev = BatchNormalization(name="sev_bn1")(sev)
    sev = Dense(64, activation="relu", name="sev_dense2")(sev)
    sev = Dropout(0.2, name="sev_dropout")(sev)
    severity_output = Dense(1, activation="sigmoid", name="severity_output")(sev)

    model = Model(inputs=inputs, outputs=[class_output, severity_output],
                  name="Corn_Ultimate_EfficientNetB0")

    # Dummy class weights for compile (only needed for loss signature, not inference)
    dummy_weights = [1.0] * NUM_CLASSES
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0),
        loss={
            "class_output": make_weighted_sparse_ce(dummy_weights),
            "severity_output": underestimation_penalty_loss,
        },
        loss_weights={"class_output": 1.0, "severity_output": 5.0},
        metrics={"class_output": "accuracy", "severity_output": "mae"},
    )
    return model


# ── Load model at startup ──
print("🔧 Building model architecture...")
model = build_model()

if not WEIGHTS_PATH.exists():
    print(f"❌ Weights not found at {WEIGHTS_PATH}")
    sys.exit(1)

model.load_weights(str(WEIGHTS_PATH))
print(f"✅ Loaded weights from {WEIGHTS_PATH}")
print("🌽 Model ready for inference!")

from flask import Flask
from flask_cors import CORS
# ── Flask App ──
app = Flask(__name__, static_folder="static")
CORS(app)


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/diseases")
def diseases_page():
    """Serve the diseases library page."""
    return send_from_directory("static", "diseases.html")


@app.route("/api/diseases")
def api_diseases():
    """Return all disease information as JSON."""
    return jsonify(CLASS_INFO)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        # Read and preprocess image
        img_bytes = file.read()
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE)
        img = preprocess_input(tf.cast(img, tf.float32))
        img_batch = tf.expand_dims(img, axis=0)

        # Predict
        class_probs, severity = model.predict(img_batch, verbose=0)

        class_id = int(np.argmax(class_probs[0]))
        class_name = CLASS_NAMES[class_id]
        confidence = float(class_probs[0][class_id]) * 100
        severity_pct = float(severity[0][0]) * 100

        # Force healthy → 0% severity
        if class_name == "healthy":
            severity_pct = 0.0

        # Severity level
        if severity_pct < 15:
            sev_level = "Low"
        elif severity_pct < 40:
            sev_level = "Moderate"
        elif severity_pct < 70:
            sev_level = "High"
        else:
            sev_level = "Severe"

        # All class probabilities
        all_probs = {
            CLASS_NAMES[i]: round(float(class_probs[0][i]) * 100, 2)
            for i in range(NUM_CLASSES)
        }

        info = CLASS_INFO[class_name]

        result = {
            "class_name": class_name,
            "display_name": info["display"],
            "emoji": info["emoji"],
            "color": info["color"],
            "confidence": round(confidence, 1),
            "severity_pct": round(severity_pct, 1),
            "severity_level": sev_level,
            "description": info["description"],
            "treatment": info["treatment"],
            "all_probs": all_probs,
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _predict_single_image(img_bytes):
    """Internal: predict a single image from bytes. Returns result dict."""
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = preprocess_input(tf.cast(img, tf.float32))
    img_batch = tf.expand_dims(img, axis=0)

    class_probs, severity = model.predict(img_batch, verbose=0)

    class_id = int(np.argmax(class_probs[0]))
    class_name = CLASS_NAMES[class_id]
    confidence = float(class_probs[0][class_id]) * 100
    severity_pct = float(severity[0][0]) * 100

    if class_name == "healthy":
        severity_pct = 0.0

    if severity_pct < 15:
        sev_level = "Low"
    elif severity_pct < 40:
        sev_level = "Moderate"
    elif severity_pct < 70:
        sev_level = "High"
    else:
        sev_level = "Severe"

    all_probs = {
        CLASS_NAMES[i]: round(float(class_probs[0][i]) * 100, 2)
        for i in range(NUM_CLASSES)
    }

    info = CLASS_INFO[class_name]

    return {
        "class_name": class_name,
        "display_name": info["display"],
        "emoji": info["emoji"],
        "color": info["color"],
        "confidence": round(confidence, 1),
        "severity_pct": round(severity_pct, 1),
        "severity_level": sev_level,
        "description": info["description"],
        "treatment": info["treatment"],
        "all_probs": all_probs,
    }


@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    """Predict multiple images at once. Accepts 'images' as multiple files."""
    files = request.files.getlist("images")
    if not files or len(files) == 0:
        return jsonify({"error": "No images uploaded"}), 400

    results = []
    errors = []

    for i, file in enumerate(files):
        if file.filename == "":
            continue
        try:
            img_bytes = file.read()
            result = _predict_single_image(img_bytes)
            result["filename"] = file.filename
            result["index"] = i
            results.append(result)
        except Exception as e:
            errors.append({"filename": file.filename, "index": i, "error": str(e)})

    # Summary statistics
    summary = {
        "total": len(results),
        "errors": len(errors),
        "disease_counts": {},
        "avg_severity": 0.0,
    }
    if results:
        for r in results:
            name = r["display_name"]
            summary["disease_counts"][name] = summary["disease_counts"].get(name, 0) + 1
        diseased = [r for r in results if r["class_name"] != "healthy"]
        if diseased:
            summary["avg_severity"] = round(
                sum(r["severity_pct"] for r in diseased) / len(diseased), 1
            )

    return jsonify({"results": results, "errors": errors, "summary": summary})


# SPATIAL ANALYSIS: Multi-disease detection via patch-based classification

GRID_SIZE = 4  # 4×4 = 16 patches per image

CLASS_COLORS = {
    "common_rust": "#FF6B35",
    "gray_leaf_spot": "#78909C",
    "healthy": "#43A047",
    "northern_leaf_blight": "#FFA726",
}


def _analyze_patches(img_tensor, grid_size=GRID_SIZE):
    """
    Divide an image into grid_size × grid_size patches, classify each.
    Returns per-patch results and aggregated disease breakdown.
    """
    img_h = tf.shape(img_tensor)[0]
    img_w = tf.shape(img_tensor)[1]

    # Resize to a larger working size for better patch quality
    work_size = IMG_SIZE[0] * 2  # 512×512
    img_large = tf.image.resize(img_tensor, (work_size, work_size))

    patch_h = work_size // grid_size
    patch_w = work_size // grid_size

    patches_batch = []
    positions = []

    for row in range(grid_size):
        for col in range(grid_size):
            y1 = row * patch_h
            x1 = col * patch_w
            patch = img_large[y1:y1 + patch_h, x1:x1 + patch_w, :]
            # Resize patch to model input size
            patch_resized = tf.image.resize(patch, IMG_SIZE)
            patch_preprocessed = preprocess_input(tf.cast(patch_resized, tf.float32))
            patches_batch.append(patch_preprocessed)
            positions.append((row, col))

    # Batch predict all patches at once (faster than one-by-one)
    patches_tensor = tf.stack(patches_batch, axis=0)  # (16, 256, 256, 3)
    class_probs_all, severity_all = model.predict(patches_tensor, verbose=0)

    # Build per-patch results
    patch_results = []
    for idx, (row, col) in enumerate(positions):
        class_id = int(np.argmax(class_probs_all[idx]))
        class_name = CLASS_NAMES[class_id]
        confidence = float(class_probs_all[idx][class_id]) * 100
        sev_pct = float(severity_all[idx][0]) * 100 if class_name != "healthy" else 0.0

        patch_results.append({
            "row": row,
            "col": col,
            "class_name": class_name,
            "display_name": CLASS_INFO[class_name]["display"],
            "emoji": CLASS_INFO[class_name]["emoji"],
            "color": CLASS_COLORS.get(class_name, "#888"),
            "confidence": round(confidence, 1),
            "severity_pct": round(sev_pct, 1),
        })

    # Aggregate: per-disease area breakdown
    total_patches = len(patch_results)
    disease_breakdown = {}

    for name in CLASS_NAMES:
        matching = [p for p in patch_results if p["class_name"] == name]
        count = len(matching)
        area_pct = (count / total_patches) * 100
        avg_sev = (
            sum(p["severity_pct"] for p in matching) / count
            if count > 0
            else 0.0
        )
        avg_conf = (
            sum(p["confidence"] for p in matching) / count
            if count > 0
            else 0.0
        )

        if count > 0:
            info = CLASS_INFO[name]
            disease_breakdown[name] = {
                "display_name": info["display"],
                "emoji": info["emoji"],
                "color": CLASS_COLORS.get(name, "#888"),
                "area_percentage": round(area_pct, 1),
                "patch_count": count,
                "avg_severity": round(avg_sev, 1),
                "avg_confidence": round(avg_conf, 1),
                "description": info["description"],
                "treatment": info["treatment"],
            }

    # Overall infection %
    infected_patches = sum(1 for p in patch_results if p["class_name"] != "healthy")
    overall_infection = (infected_patches / total_patches) * 100

    # Number of distinct diseases found
    diseases_found = [
        name for name in CLASS_NAMES
        if name != "healthy" and name in disease_breakdown
    ]

    return {
        "grid_size": grid_size,
        "total_patches": total_patches,
        "overall_infection_pct": round(overall_infection, 1),
        "healthy_pct": round(100 - overall_infection, 1),
        "diseases_found": len(diseases_found),
        "disease_names": [CLASS_INFO[d]["display"] for d in diseases_found],
        "disease_breakdown": disease_breakdown,
        "patch_grid": patch_results,
    }


@app.route("/predict-spatial", methods=["POST"])
def predict_spatial():
    """
    Spatial analysis: divide a single image into patches, classify each,
    and return per-disease area breakdown.
    
    Use case: when a single leaf has multiple diseases, this shows what
    percentage of the image is affected by each disease.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        img_bytes = file.read()
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)

        # Also get the full-image prediction for comparison
        img_resized = tf.image.resize(img, IMG_SIZE)
        img_preprocessed = preprocess_input(tf.cast(img_resized, tf.float32))
        img_batch = tf.expand_dims(img_preprocessed, axis=0)
        full_class_probs, full_severity = model.predict(img_batch, verbose=0)

        full_class_id = int(np.argmax(full_class_probs[0]))
        full_class_name = CLASS_NAMES[full_class_id]
        full_confidence = float(full_class_probs[0][full_class_id]) * 100
        full_severity_pct = float(full_severity[0][0]) * 100
        if full_class_name == "healthy":
            full_severity_pct = 0.0

        full_info = CLASS_INFO[full_class_name]

        # Run spatial analysis
        grid_size = int(request.form.get("grid_size", GRID_SIZE))
        grid_size = max(2, min(grid_size, 8))  # clamp to 2-8
        spatial = _analyze_patches(img, grid_size=grid_size)

        result = {
            # Full-image prediction
            "full_image": {
                "class_name": full_class_name,
                "display_name": full_info["display"],
                "emoji": full_info["emoji"],
                "color": CLASS_COLORS.get(full_class_name, "#888"),
                "confidence": round(full_confidence, 1),
                "severity_pct": round(full_severity_pct, 1),
                "all_probs": {
                    CLASS_NAMES[i]: round(float(full_class_probs[0][i]) * 100, 2)
                    for i in range(NUM_CLASSES)
                },
            },
            # Spatial breakdown
            "spatial": spatial,
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n🌐 Starting server at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
