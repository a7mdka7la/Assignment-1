"""Data pipeline for Medical MNIST.

Handles Google Drive mounting, Kaggle download, extraction, and the per-class
tf.data pipeline used by both AE and VAE training.

The Kaggle dataset identifier is ``andrewmvd/medical-mnist``. After extraction
the directory layout is::

    dataset_dir/
        AbdomenCT/
        BreastMRI/
        CXR/
        ChestCT/
        Hand/
        HeadCT/
"""

from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from typing import Dict, List, Tuple

import tensorflow as tf

from .config import (
    CLASS_NAMES,
    IMG_SIZE,
    NUM_CHANNELS,
    PATHS,
    SEED,
    TRAIN,
)


AUTOTUNE = tf.data.AUTOTUNE


# --------------------------------------------------------------------------- #
# Colab / Drive / Kaggle plumbing
# --------------------------------------------------------------------------- #

def mount_drive() -> None:
    """Mount Google Drive inside Colab. No-op when not running in Colab."""
    try:
        from google.colab import drive  # type: ignore
    except ImportError:
        print("Not running in Colab; skipping Drive mount.")
        return
    drive.mount("/content/drive", force_remount=False)


def install_kaggle_credentials(kaggle_json_path: str = PATHS.kaggle_json) -> None:
    """Copy kaggle.json from Drive into ~/.kaggle with the right permissions."""
    target_dir = os.path.expanduser("~/.kaggle")
    target_file = os.path.join(target_dir, "kaggle.json")
    os.makedirs(target_dir, exist_ok=True)
    if not os.path.exists(kaggle_json_path):
        raise FileNotFoundError(
            f"kaggle.json not found at {kaggle_json_path}. "
            "Upload it to that Drive path or override PATHS.kaggle_json."
        )
    shutil.copyfile(kaggle_json_path, target_file)
    os.chmod(target_file, 0o600)


def download_dataset(
    dataset_slug: str = "andrewmvd/medical-mnist",
    dest_zip: str = PATHS.dataset_zip,
) -> str:
    """Download the Kaggle zip to Drive if it isn't already there."""
    if os.path.exists(dest_zip):
        print(f"Dataset zip already present at {dest_zip}; skipping download.")
        return dest_zip
    os.makedirs(os.path.dirname(dest_zip), exist_ok=True)
    subprocess.run(["pip", "install", "-q", "kaggle"], check=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", "/tmp",
         "--force"],
        check=True,
    )
    tmp_zip = f"/tmp/{dataset_slug.split('/')[-1]}.zip"
    shutil.move(tmp_zip, dest_zip)
    return dest_zip


def ensure_dataset(
    dataset_zip: str = PATHS.dataset_zip,
    extract_dir: str = PATHS.dataset_dir,
) -> str:
    """Extract the dataset zip once and return the root dataset directory.

    The archive contains a single top-level folder. We return the path that
    directly contains the six class folders.
    """
    if not _looks_extracted(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
        print(f"Extracting {dataset_zip} -> {extract_dir} ...")
        with zipfile.ZipFile(dataset_zip) as zf:
            zf.extractall(extract_dir)

    root = _find_class_root(extract_dir)
    counts = {c: len(list_class_files(c, root)) for c in CLASS_NAMES}
    print("Per-class image counts:")
    for c, n in counts.items():
        print(f"  {c:<12} {n}")
        if n < 1000:
            print(f"  WARNING: {c} has fewer than 1000 images.")
    return root


def _looks_extracted(extract_dir: str) -> bool:
    if not os.path.isdir(extract_dir):
        return False
    try:
        root = _find_class_root(extract_dir)
    except FileNotFoundError:
        return False
    return all(
        os.path.isdir(os.path.join(root, c)) for c in CLASS_NAMES
    )


def _find_class_root(extract_dir: str) -> str:
    """Return the directory that directly contains the 6 class folders."""
    if all(os.path.isdir(os.path.join(extract_dir, c)) for c in CLASS_NAMES):
        return extract_dir
    for entry in sorted(os.listdir(extract_dir)):
        candidate = os.path.join(extract_dir, entry)
        if os.path.isdir(candidate) and all(
            os.path.isdir(os.path.join(candidate, c)) for c in CLASS_NAMES
        ):
            return candidate
    raise FileNotFoundError(
        f"Could not locate the 6 class folders under {extract_dir}."
    )


# --------------------------------------------------------------------------- #
# File listing and splitting
# --------------------------------------------------------------------------- #

def list_class_files(class_name: str, root: str) -> List[str]:
    """Return sorted PNG paths for one anatomical region."""
    class_dir = os.path.join(root, class_name)
    if not os.path.isdir(class_dir):
        raise FileNotFoundError(f"Class folder missing: {class_dir}")
    files = [
        os.path.join(class_dir, f)
        for f in sorted(os.listdir(class_dir))
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    return files


def split_files(
    files: List[str],
    val_fraction: float = TRAIN.val_fraction,
    test_fraction: float = TRAIN.test_fraction,
    seed: int = SEED,
) -> Tuple[List[str], List[str], List[str]]:
    """Deterministic train/val/test split of a list of file paths."""
    rng = tf.random.Generator.from_seed(seed)
    idx = tf.argsort(rng.uniform([len(files)])).numpy()
    shuffled = [files[i] for i in idx]
    n = len(shuffled)
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))
    test = shuffled[:n_test]
    val = shuffled[n_test:n_test + n_val]
    train = shuffled[n_test + n_val:]
    return train, val, test


# --------------------------------------------------------------------------- #
# tf.data pipelines
# --------------------------------------------------------------------------- #

def _decode_image(path: tf.Tensor) -> tf.Tensor:
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=NUM_CHANNELS, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape((IMG_SIZE, IMG_SIZE, NUM_CHANNELS))
    return img


def build_dataset(
    paths: List[str],
    batch_size: int = TRAIN.batch_size,
    shuffle: bool = True,
    seed: int = SEED,
) -> tf.data.Dataset:
    """Build an unlabeled tf.data pipeline yielding image tensors."""
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(_decode_image, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(TRAIN.shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def build_labeled_dataset(
    paths: List[str],
    labels: List[int],
    batch_size: int = TRAIN.batch_size,
    shuffle: bool = False,
    seed: int = SEED,
) -> tf.data.Dataset:
    """Build a labeled pipeline for latent-space visualization."""
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(
        lambda p, y: (_decode_image(p), y),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(TRAIN.shuffle_buffer, seed=seed)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


# --------------------------------------------------------------------------- #
# Higher-level dataset builders
# --------------------------------------------------------------------------- #

def per_class_splits(root: str) -> Dict[str, Tuple[List[str], List[str], List[str]]]:
    """Train/val/test file splits for each class."""
    splits: Dict[str, Tuple[List[str], List[str], List[str]]] = {}
    for c in CLASS_NAMES:
        files = list_class_files(c, root)
        splits[c] = split_files(files)
    return splits


def global_labeled_split(
    root: str,
) -> Tuple[
    Tuple[List[str], List[int]],
    Tuple[List[str], List[int]],
    Tuple[List[str], List[int]],
]:
    """Combine all classes into one labeled train/val/test split.

    Returns three (paths, labels) tuples. Labels are indices into CLASS_NAMES.
    """
    train_p, train_y = [], []
    val_p, val_y = [], []
    test_p, test_y = [], []
    for label, c in enumerate(CLASS_NAMES):
        tr, va, te = split_files(list_class_files(c, root))
        train_p.extend(tr); train_y.extend([label] * len(tr))
        val_p.extend(va);   val_y.extend([label] * len(va))
        test_p.extend(te);  test_y.extend([label] * len(te))
    return (train_p, train_y), (val_p, val_y), (test_p, test_y)
