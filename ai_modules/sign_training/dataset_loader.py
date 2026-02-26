"""ASL Alphabet dataset loader (A–Z only) using MediaPipe hand landmarks.

Constraints implemented:
- Load from `datasets/asl_alphabet_train/asl_alphabet_train`
- Use A–Z only
- Ignore: nothing/space/delete/del
- Limit images per class to 700
- Extract 21 hand landmarks => 63 features per sample (x,y,z)

Returns:
- X: np.ndarray shape (N, 63)
- y: np.ndarray shape (N,) integer labels
- label_map: dict[str, int]
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


class DatasetLoadError(RuntimeError):
    pass


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _is_az_folder(name: str) -> bool:
    n = name.strip()
    return len(n) == 1 and ("A" <= n <= "Z" or "a" <= n <= "z")


def _resolve_default_dataset_dir() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "datasets" / "asl_alphabet_train" / "asl_alphabet_train"


def _list_images(folder: Path) -> List[Path]:
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS]


def _extract_landmarks_vector(img_path: Path, hands) -> Optional[np.ndarray]:
    try:
        import cv2
    except Exception as e:
        raise DatasetLoadError("Missing dependency: opencv-python") from e

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None

    # MediaPipe is much faster on smaller images; landmarks are normalized so
    # resizing doesn't affect coordinate meaning.
    h, w = img_bgr.shape[:2]
    max_dim = max(h, w)
    if max_dim > 320:
        scale = 320.0 / float(max_dim)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = hands.process(img_rgb)
    if not res.multi_hand_landmarks:
        return None

    hand = res.multi_hand_landmarks[0]
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32)
    if pts.shape != (21, 3):
        return None
    return pts.reshape(-1)


def load_asl_alphabet(
    dataset_dir: Optional[Path] = None,
    *,
    excluded_classes: Sequence[str] = ("nothing", "space", "delete", "del"),
    max_per_class: int = 700,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Load ASL Alphabet training images into landmark vectors.

    Args:
        dataset_dir: directory containing class folders A..Z.
        excluded_classes: folder names to ignore.
        max_per_class: cap the number of valid landmark samples per class.
        seed: RNG seed for deterministic sampling.
    """

    root = dataset_dir or _resolve_default_dataset_dir()
    if not root.exists():
        raise DatasetLoadError(f"Dataset directory not found: {root}")

    excluded = {c.strip().lower() for c in excluded_classes}
    class_dirs = [p for p in root.iterdir() if p.is_dir()]
    class_names = []
    for d in class_dirs:
        name = d.name.strip()
        if name.lower() in excluded:
            continue
        if not _is_az_folder(name):
            continue
        class_names.append(name.upper())

    class_names = sorted(set(class_names))
    if len(class_names) != 26:
        # Still proceed, but training will have fewer classes.
        pass

    label_map: Dict[str, int] = {name: i for i, name in enumerate(class_names)}

    try:
        import mediapipe as mp
    except Exception as e:
        raise DatasetLoadError("Missing dependency: mediapipe") from e

    rng = np.random.default_rng(seed)

    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    for cls in class_names:
        cls_dir = root / cls
        if not cls_dir.exists():
            # Some datasets use lowercase folders.
            cls_dir = root / cls.lower()
        if not cls_dir.exists():
            continue

        images = _list_images(cls_dir)
        if not images:
            continue
        print(f"[{cls}] images={len(images)} target={max_per_class}", flush=True)
        rng.shuffle(images)

        collected = 0
        scanned = 0

        # Using a single MediaPipe instance is much faster on Windows than
        # spawning subprocesses. To reduce the chance of rare long-run stalls,
        # restart the graph periodically.
        restart_every = 400
        processed_since_restart = restart_every
        hands = None
        
        def _make_hands():
            return mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                model_complexity=0,
                min_detection_confidence=0.5,
            )

        try:
            for img_path in images:
                if collected >= max_per_class:
                    break

                if hands is None or processed_since_restart >= restart_every:
                    if hands is not None:
                        try:
                            hands.close()
                        except Exception:
                            pass
                    hands = _make_hands()
                    processed_since_restart = 0

                scanned += 1
                processed_since_restart += 1

                try:
                    vec = _extract_landmarks_vector(img_path, hands)
                except Exception:
                    # If MediaPipe or OpenCV throws, restart the graph and keep going.
                    try:
                        hands.close()
                    except Exception:
                        pass
                    hands = _make_hands()
                    processed_since_restart = 0
                    continue

                if vec is None or vec.shape != (63,):
                    continue

                X_list.append(vec.astype(np.float32, copy=False))
                y_list.append(int(label_map[cls]))
                collected += 1

                if scanned % 250 == 0:
                    print(f"[{cls}] progress scanned={scanned} collected={collected}", flush=True)
        finally:
            if hands is not None:
                try:
                    hands.close()
                except Exception:
                    pass

        print(f"[{cls}] scanned={scanned} collected={collected}", flush=True)

    if not X_list:
        raise DatasetLoadError("No landmark samples extracted. Check dataset path and dependencies.")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y, label_map
