"""Train ASL Alphabet (A–Z) landmark classifier and save Keras model.

Runnable:
    python -m ai_modules.sign_training.train
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .dataset_loader import load_asl_alphabet
from .evaluate import evaluate
from .preprocess import preprocess


def build_model(input_dim: int, num_classes: int):
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = project_root / "datasets" / "asl_alphabet_train" / "asl_alphabet_train"
    model_path = project_root / "models" / "sign_model.h5"
    metrics_path = project_root / "models" / "sign_model_metrics.json"

    # Fail fast: don't spend minutes extracting landmarks if TensorFlow isn't usable.
    try:
        import tensorflow as tf
    except Exception as e:
        raise RuntimeError(
            "TensorFlow is not installed correctly in this environment. "
            "Fix TensorFlow first, then rerun training."
        ) from e

    print(f"TensorFlow version: {tf.__version__}", flush=True)

    print(f"Loading dataset from: {dataset_dir}", flush=True)

    X, y, label_map = load_asl_alphabet(
        dataset_dir=dataset_dir,
        excluded_classes=("nothing", "space", "delete", "del"),
        max_per_class=700,
        seed=42,
    )

    print(f"Extracted samples: {int(X.shape[0])} | classes: {len(label_map)}", flush=True)

    X_train, X_test, y_train, y_test = preprocess(X, y, test_ratio=0.2, seed=42)
    num_classes = int(y_train.shape[1])

    print(
        f"Split: train={int(X_train.shape[0])} test={int(X_test.shape[0])} | num_classes={num_classes}",
        flush=True,
    )

    tf.random.set_seed(42)
    np.random.seed(42)

    model = build_model(input_dim=63, num_classes=num_classes)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        )
    ]

    print("Training...", flush=True)

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=40,
        batch_size=64,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1).astype(np.int32)
    metrics = evaluate(y_test, y_pred, label_map=label_map)

    print(f"Test accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}", flush=True)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    print(f"Saved model: {model_path}", flush=True)

    report: Dict[str, Any] = {
        "model_path": str(Path("models") / "sign_model.h5"),
        "metrics_path": str(Path("models") / "sign_model_metrics.json"),
        "dataset_dir": str(Path("datasets") / "asl_alphabet_train" / "asl_alphabet_train"),
        "num_samples": int(X.shape[0]),
        "num_classes": int(num_classes),
        "label_map": label_map,
        "test_accuracy": metrics["accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "per_class_f1": metrics["per_class_f1"],
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
    }

    import json

    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved metrics: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
