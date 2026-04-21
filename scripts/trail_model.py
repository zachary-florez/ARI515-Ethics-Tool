"""
train_model.py — Satellite AI-Image Detection
==============================================
Trains a MobileNetV2-based binary classifier on your 2000 satellite images
and exports a TensorFlow SavedModel that ModelTrainer.java can load.

Requirements:
    pip install tensorflow pillow scikit-learn

Usage:
    python train_model.py \
        --data_dir  ./data/raw \
        --csv       ./data/metadata.csv \
        --output    ./models/satellite_detector \
        --epochs    20 \
        --img_size  224

The CSV must have columns: filename, label  (label = "real" or "ai_generated")
"""

import argparse
import os
import csv
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="./data/raw")
    p.add_argument("--csv",      default="./data/metadata.csv")
    p.add_argument("--output",   default="./models/satellite_detector")
    p.add_argument("--epochs",   type=int, default=20)
    p.add_argument("--batch",    type=int, default=32)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--lr",       type=float, default=1e-4)
    return p.parse_args()

LABEL_MAP = {"real": 0, "ai_generated": 1}

def load_dataset(data_dir: str, csv_path: str, img_size: int):
    """Reads the metadata CSV and loads images into numpy arrays."""
    images, labels = [], []
    skipped = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loading {len(rows)} entries from CSV...")

    for row in rows:
        img_path = Path(data_dir) / row["filename"]
        label_str = row["label"].strip().lower()

        if label_str not in LABEL_MAP:
            print(f"  Skipping unknown label '{label_str}' for {row['filename']}")
            skipped += 1
            continue

        if not img_path.exists():
            print(f"  Skipping missing file: {img_path}")
            skipped += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((img_size, img_size), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0   # normalise to [0, 1]
            images.append(arr)
            labels.append(LABEL_MAP[label_str])
        except Exception as e:
            print(f"  Error loading {img_path}: {e}")
            skipped += 1

    print(f"Loaded {len(images)} images ({skipped} skipped).")
    return np.stack(images), np.array(labels, dtype=np.int32)

def build_model(img_size: int, learning_rate: float) -> Model:
    """
    MobileNetV2 backbone (pretrained on ImageNet) + custom classifier head.
    Transfer learning works well even with only 2000 images.
    Data augmentation added to improve generalization on real images.
    """
    base = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet"
    )
    # Freeze backbone initially; we fine-tune later
    base.trainable = False

    # Data augmentation — helps model generalize on real images
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ], name="data_augmentation")

    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="input_image")
    x = data_augmentation(inputs, training=True)  # ← augment input
    x = base(x, training=False)                   # ← pass augmented to backbone
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    # Single sigmoid output — probability of being AI-generated
    outputs = layers.Dense(1, activation="sigmoid", name="ai_probability")(x)

    model = Model(inputs, outputs, name="satellite_detector")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model

def fine_tune(model: Model, learning_rate: float, unfreeze_from: int = 100):
    """Unfreezes the top layers of MobileNetV2 for a second training pass."""
    base = model.layers[1]          # MobileNetV2 is the second layer
    base.trainable = True
    for layer in base.layers[:unfreeze_from]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate / 10),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    print(f"Fine-tuning: {sum(1 for l in base.layers if l.trainable)} backbone layers unfrozen.")

def export_savedmodel(model: Model, output_dir: str):
    """
    Saves in TF SavedModel format.
    Java loads this with: SavedModelBundle.load(output_dir, "serve")

    The serving signature will have:
      INPUT  'serving_default_input_image'   (shape [None, H, W, 3])
      OUTPUT 'StatefulPartitionedCall'       (shape [None, 1])

    Run:  saved_model_cli show --dir <output_dir> --all
    to confirm the exact tensor names to pass to ModelTrainer.smokeTest().
    """
    os.makedirs(output_dir, exist_ok=True)
    model.export(output_dir)
    print(f"\nSavedModel written to: {output_dir}")
    print("Load in Java with:  SavedModelBundle.load(\"" + output_dir + "\", \"serve\")")

def main():
    args = parse_args()

    # 1. Load data
    X, y = load_dataset(args.data_dir, args.csv, args.img_size)
    print(f"Class balance — real: {(y==0).sum()}, ai_generated: {(y==1).sum()}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # 2. Build model
    model = build_model(args.img_size, args.lr)
    model.summary()

    # 3. Phase 1 — train head only
    print("\n=== Phase 1: Training classifier head ===")
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_auc", mode="max"),
        ModelCheckpoint("./checkpoints/best_head.keras", save_best_only=True, monitor="val_auc", mode="max"),
    ]
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=callbacks,
    )

    # 4. Phase 2 — fine-tune top backbone layers
    print("\n=== Phase 2: Fine-tuning backbone ===")
    fine_tune(model, args.lr)
    callbacks[1] = ModelCheckpoint("./checkpoints/best_finetune.keras", save_best_only=True,
                                   monitor="val_auc", mode="max")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs // 2,
        batch_size=args.batch,
        callbacks=callbacks,
    )

    # 5. Final validation metrics
    print("\n=== Final evaluation on validation set ===")
    results = model.evaluate(X_val, y_val, batch_size=args.batch)
    for name, val in zip(model.metrics_names, results):
        print(f"  {name}: {val:.4f}")

    # 6. Export
    export_savedmodel(model, args.output)


if __name__ == "__main__":
    main()