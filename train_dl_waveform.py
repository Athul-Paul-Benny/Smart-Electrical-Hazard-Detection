"""
Optional Deep Learning Benchmark (1D-CNN on Raw 5 kHz Waveforms)
AI-Based Electrical Hazard & Arc-Fault Detection System

Loads 'raw_waveforms_single_phase.npz', stacks Line + Neutral CT waveforms
into a 2-channel 1D input tensor shape (1050, 2500, 2), and trains a 1D-CNN.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

import tensorflow as tf
from tensorflow.keras import layers, models


def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_raw_waveforms(npz_path="raw_waveforms_single_phase.npz"):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Waveform file not found at {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    line = data["line"]        # shape: (1050, 2500)
    neutral = data["neutral"]  # shape: (1050, 2500)
    labels = data["labels"]    # shape: (1050,)

    # Stack line and neutral into dual-channel 1D tensor (samples, time_steps, channels)
    X = np.stack([line, neutral], axis=-1)
    
    le = LabelEncoder()
    y = le.fit_transform(labels)
    classes = list(le.classes_)

    print(f"[INFO] Loaded raw waveforms with shape: {X.shape}")
    print(f"[INFO] Classes ({len(classes)}): {classes}")

    return X, y, classes, le


def build_1d_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Conv1D(filters=32, kernel_size=15, strides=2, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        
        layers.Conv1D(filters=64, kernel_size=7, strides=1, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        
        layers.Conv1D(filters=128, kernel_size=5, strides=1, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    set_seed(42)
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    X, y, classes, le = load_raw_waveforms()

    # Split 70% Train, 15% Val, 15% Test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    val_ratio = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=42
    )

    print(f"[INFO] Waveform Train shape: {X_train.shape}")
    print(f"[INFO] Waveform Val shape:   {X_val.shape}")
    print(f"[INFO] Waveform Test shape:  {X_test.shape}")

    # Standardize channels per sample
    mean = np.mean(X_train, axis=(0, 1), keepdims=True)
    std = np.std(X_train, axis=(0, 1), keepdims=True) + 1e-8

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    cnn_model = build_1d_cnn(input_shape=X_train.shape[1:], num_classes=len(classes))
    cnn_model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ]

    print("\n[INFO] Training 1D-CNN on raw 5 kHz waveforms...")
    history = cnn_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    test_preds_prob = cnn_model.predict(X_test)
    test_preds = np.argmax(test_preds_prob, axis=1)

    acc = accuracy_score(y_test, test_preds)
    f1_m = f1_score(y_test, test_preds, average="macro")

    print("\n=======================================================")
    print(" 1D-CNN WAVEFORM MODEL TEST RESULTS")
    print("=======================================================")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Macro F1: {f1_m:.4f}")
    print("\nClassification Report (1D-CNN Raw Waveforms):")
    print(classification_report(y_test, test_preds, target_names=classes, digits=4))

    dl_results = {
        "model": "1D-CNN (Raw Waveforms)",
        "input_shape": list(X_train.shape[1:]),
        "test_accuracy": float(acc),
        "test_macro_f1": float(f1_m)
    }

    with open("results/dl_metrics.json", "w") as f:
        json.dump(dl_results, f, indent=4)
    
    cnn_model.save("models/cnn_waveform_model.keras")
    print("[INFO] Saved 1D-CNN model to 'models/cnn_waveform_model.keras'")
    print("[INFO] Deep learning waveform benchmark complete!")


if __name__ == "__main__":
    main()
