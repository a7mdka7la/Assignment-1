"""Training loops for per-class and global AE/VAE runs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import tensorflow as tf

from .config import CLASS_NAMES, PATHS, SEED, TRAIN
from .data import (
    build_dataset,
    build_labeled_dataset,
    global_labeled_split,
    per_class_splits,
)
from .models.ae import make_autoencoder
from .models.vae import KLWarmup, make_vae


ModelFactory = Callable[[], tf.keras.Model]


@dataclass
class RunResult:
    tag: str
    model: tf.keras.Model
    history: tf.keras.callbacks.History
    weights_path: str


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _checkpoint_callbacks(tag: str, monitor: str) -> List[tf.keras.callbacks.Callback]:
    _ensure_dir(PATHS.checkpoints)
    path = os.path.join(PATHS.checkpoints, f"{tag}.weights.h5")
    cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=path,
        save_weights_only=True,
        save_best_only=True,
        monitor=monitor,
        mode="min",
        verbose=0,
    )
    return [cb]


def train_one(
    model_factory: ModelFactory,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    tag: str,
    epochs: int = TRAIN.epochs,
    extra_callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
) -> RunResult:
    """Train a freshly built model and return its history + checkpoint path."""
    model = model_factory()
    monitor = "val_loss"
    callbacks = _checkpoint_callbacks(tag, monitor)
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    print(f"\n[train_one] tag={tag} epochs={epochs}")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )
    weights_path = os.path.join(PATHS.checkpoints, f"{tag}.weights.h5")
    model.load_weights(weights_path)
    return RunResult(tag=tag, model=model, history=history, weights_path=weights_path)


def _ae_factory() -> ModelFactory:
    return lambda: make_autoencoder()


def _vae_factory() -> ModelFactory:
    return lambda: make_vae()


def train_per_class(
    root: str,
    epochs: int = TRAIN.epochs,
    batch_size: int = TRAIN.batch_size,
) -> Dict[str, RunResult]:
    """Train one AE and one VAE for each of the 6 classes."""
    splits = per_class_splits(root)
    results: Dict[str, RunResult] = {}

    for class_name in CLASS_NAMES:
        train_files, val_files, _ = splits[class_name]
        train_ds = build_dataset(train_files, batch_size=batch_size, shuffle=True, seed=SEED)
        val_ds = build_dataset(val_files, batch_size=batch_size, shuffle=False)

        ae_tag = f"ae_{class_name}"
        results[ae_tag] = train_one(
            _ae_factory(), train_ds, val_ds, ae_tag, epochs=epochs,
        )

        vae_tag = f"vae_{class_name}"
        results[vae_tag] = train_one(
            _vae_factory(), train_ds, val_ds, vae_tag, epochs=epochs,
            extra_callbacks=[KLWarmup(TRAIN.kl_warmup_epochs)],
        )

    return results


def train_global(
    root: str,
    epochs: int = TRAIN.epochs,
    batch_size: int = TRAIN.batch_size,
) -> Tuple[Dict[str, RunResult], Dict[str, tf.data.Dataset]]:
    """Train a global AE and a global VAE on all classes combined.

    Returns the trained runs plus labeled train/val/test datasets suitable for
    the cross-class latent visualization.
    """
    (train_p, train_y), (val_p, val_y), (test_p, test_y) = global_labeled_split(root)

    train_unlabeled = build_dataset(train_p, batch_size=batch_size, shuffle=True, seed=SEED)
    val_unlabeled = build_dataset(val_p, batch_size=batch_size, shuffle=False)

    results: Dict[str, RunResult] = {}
    results["ae_global"] = train_one(
        _ae_factory(), train_unlabeled, val_unlabeled, "ae_global", epochs=epochs,
    )
    results["vae_global"] = train_one(
        _vae_factory(), train_unlabeled, val_unlabeled, "vae_global", epochs=epochs,
        extra_callbacks=[KLWarmup(TRAIN.kl_warmup_epochs)],
    )

    labeled = {
        "train": build_labeled_dataset(train_p, train_y, batch_size=batch_size),
        "val": build_labeled_dataset(val_p, val_y, batch_size=batch_size),
        "test": build_labeled_dataset(test_p, test_y, batch_size=batch_size),
    }
    return results, labeled


def train_all(
    root: str,
    epochs: int = TRAIN.epochs,
    batch_size: int = TRAIN.batch_size,
) -> Tuple[Dict[str, RunResult], Dict[str, tf.data.Dataset]]:
    """Run the full 14-model training sweep and return all runs + labeled data."""
    per_class = train_per_class(root, epochs=epochs, batch_size=batch_size)
    globals_, labeled = train_global(root, epochs=epochs, batch_size=batch_size)
    all_runs = {**per_class, **globals_}
    return all_runs, labeled
