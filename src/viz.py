"""Plotting helpers. Every function saves its figure to ``figures/``."""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .config import CLASS_NAMES, LATENT_DIM, PATHS
from .utils import add_gaussian_noise


def _ensure_figures_dir() -> None:
    os.makedirs(PATHS.figures, exist_ok=True)


def _savefig(fig: plt.Figure, name: str) -> str:
    _ensure_figures_dir()
    path = os.path.join(PATHS.figures, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Loss curves
# --------------------------------------------------------------------------- #

def plot_loss_curves(histories: Dict[str, tf.keras.callbacks.History],
                     filename: str = "loss_curves.png") -> str:
    """Train/val loss per run, grouped by model type."""
    ae_items = [(k, v) for k, v in histories.items() if k.startswith("ae_")]
    vae_items = [(k, v) for k, v in histories.items() if k.startswith("vae_")]

    n = max(len(ae_items), len(vae_items))
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6), squeeze=False)

    for col, (tag, hist) in enumerate(ae_items):
        ax = axes[0, col]
        ax.plot(hist.history["loss"], label="train")
        ax.plot(hist.history.get("val_loss", []), label="val")
        ax.set_title(tag, fontsize=9)
        ax.set_xlabel("epoch"); ax.set_ylabel("MSE")
        ax.legend(fontsize=7)

    for col, (tag, hist) in enumerate(vae_items):
        ax = axes[1, col]
        ax.plot(hist.history["loss"], label="total")
        if "recon_loss" in hist.history:
            ax.plot(hist.history["recon_loss"], label="recon")
        if "kl_loss" in hist.history:
            ax.plot(hist.history["kl_loss"], label="KL")
        ax.set_title(tag, fontsize=9)
        ax.set_xlabel("epoch")
        ax.legend(fontsize=7)

    fig.suptitle("Training losses (top: AE, bottom: VAE)", fontsize=12)
    fig.tight_layout()
    return _savefig(fig, filename)


# --------------------------------------------------------------------------- #
# Reconstructions
# --------------------------------------------------------------------------- #

def _take_batch(dataset: tf.data.Dataset, n: int) -> np.ndarray:
    for batch in dataset.take(1):
        x = batch if not isinstance(batch, (tuple, list)) else batch[0]
        return x.numpy()[:n]
    return np.zeros((0,))


def plot_reconstructions(model, dataset, n: int = 8,
                         filename: str = "reconstructions.png",
                         title: str = "Originals (top) vs. Reconstructions (bottom)") -> str:
    x = _take_batch(dataset, n)
    x_hat = model(x, training=False).numpy()
    fig, axes = plt.subplots(2, n, figsize=(1.6 * n, 3.4))
    for i in range(n):
        axes[0, i].imshow(x[i, :, :, 0], cmap="gray"); axes[0, i].axis("off")
        axes[1, i].imshow(x_hat[i, :, :, 0], cmap="gray"); axes[1, i].axis("off")
    fig.suptitle(title, fontsize=11)
    return _savefig(fig, filename)


def plot_reconstructions_grid(
    models: Dict[str, tf.keras.Model],
    datasets: Dict[str, tf.data.Dataset],
    n: int = 6,
    filename: str = "reconstructions_all_classes.png",
    title: str = "Per-class reconstructions",
) -> str:
    """Side-by-side originals + reconstructions for all 6 classes, one model type."""
    classes = list(models.keys())
    rows = len(classes) * 2
    fig, axes = plt.subplots(rows, n, figsize=(1.6 * n, 1.6 * rows))
    for r, cls in enumerate(classes):
        x = _take_batch(datasets[cls], n)
        x_hat = models[cls](x, training=False).numpy()
        for i in range(n):
            axes[2 * r, i].imshow(x[i, :, :, 0], cmap="gray"); axes[2 * r, i].axis("off")
            axes[2 * r + 1, i].imshow(x_hat[i, :, :, 0], cmap="gray"); axes[2 * r + 1, i].axis("off")
        axes[2 * r, 0].set_ylabel(f"{cls}\norig", fontsize=8)
        axes[2 * r + 1, 0].set_ylabel("recon", fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return _savefig(fig, filename)


# --------------------------------------------------------------------------- #
# Latent space
# --------------------------------------------------------------------------- #

def _collect_latents(model, labeled_dataset) -> (np.ndarray, np.ndarray):
    all_z: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    for x, y in labeled_dataset:
        z = model.encode(x).numpy()
        all_z.append(z); all_y.append(y.numpy())
    return np.concatenate(all_z, axis=0), np.concatenate(all_y, axis=0)


def plot_latent_2d(
    model,
    labeled_dataset,
    method: str = "tsne",
    filename: Optional[str] = None,
    title: Optional[str] = None,
    max_points: int = 3000,
) -> str:
    """Encode all samples, project to 2D, color by class."""
    z, y = _collect_latents(model, labeled_dataset)
    if z.shape[0] > max_points:
        idx = np.random.RandomState(0).choice(z.shape[0], max_points, replace=False)
        z, y = z[idx], y[idx]

    if method == "pca":
        from sklearn.decomposition import PCA
        z2 = PCA(n_components=2, random_state=0).fit_transform(z)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        z2 = TSNE(n_components=2, random_state=0, init="pca", learning_rate="auto").fit_transform(z)
    else:
        raise ValueError(f"Unknown projection method: {method}")

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, name in enumerate(CLASS_NAMES):
        mask = y == label
        ax.scatter(z2[mask, 0], z2[mask, 1], s=6, alpha=0.6, label=name)
    ax.set_title(title or f"Latent space ({method.upper()} projection of {LATENT_DIM}-D latent)")
    ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    ax.legend(markerscale=2, fontsize=8, loc="best")
    fig.tight_layout()
    return _savefig(fig, filename or f"latent_{method}.png")


# --------------------------------------------------------------------------- #
# VAE-only visualizations
# --------------------------------------------------------------------------- #

def plot_vae_samples(vae, n: int = 16, filename: str = "vae_samples.png",
                     title: str = "VAE samples from N(0, I)") -> str:
    imgs = vae.sample(n).numpy()
    side = int(np.ceil(np.sqrt(n)))
    fig, axes = plt.subplots(side, side, figsize=(1.6 * side, 1.6 * side))
    for i in range(side * side):
        ax = axes[i // side, i % side]
        if i < n:
            ax.imshow(imgs[i, :, :, 0], cmap="gray")
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    return _savefig(fig, filename)


def plot_latent_traversal(
    vae,
    dim: int = 0,
    value_range=(-3.0, 3.0),
    steps: int = 8,
    filename: Optional[str] = None,
) -> str:
    """Walk one latent axis through `steps` values; show decoded images."""
    lo, hi = value_range
    values = np.linspace(lo, hi, steps)
    z = np.zeros((steps, vae.latent_dim), dtype=np.float32)
    z[:, dim] = values
    imgs = vae.decode(tf.convert_to_tensor(z)).numpy()
    fig, axes = plt.subplots(1, steps, figsize=(1.6 * steps, 1.8))
    for i in range(steps):
        axes[i].imshow(imgs[i, :, :, 0], cmap="gray")
        axes[i].set_title(f"{values[i]:.1f}", fontsize=8)
        axes[i].axis("off")
    fig.suptitle(f"Latent traversal along dim {dim}", fontsize=11)
    return _savefig(fig, filename or f"latent_traversal_dim{dim}.png")


# --------------------------------------------------------------------------- #
# Denoising
# --------------------------------------------------------------------------- #

def plot_denoising(
    model,
    dataset,
    sigma: float = 0.2,
    n: int = 8,
    filename: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    x = _take_batch(dataset, n)
    x_noisy = add_gaussian_noise(tf.convert_to_tensor(x), sigma).numpy()
    x_hat = model(x_noisy, training=False).numpy()
    fig, axes = plt.subplots(3, n, figsize=(1.6 * n, 4.8))
    for i in range(n):
        axes[0, i].imshow(x[i, :, :, 0], cmap="gray"); axes[0, i].axis("off")
        axes[1, i].imshow(x_noisy[i, :, :, 0], cmap="gray"); axes[1, i].axis("off")
        axes[2, i].imshow(x_hat[i, :, :, 0], cmap="gray"); axes[2, i].axis("off")
    axes[0, 0].set_ylabel("clean", fontsize=9)
    axes[1, 0].set_ylabel(f"noisy σ={sigma}", fontsize=9)
    axes[2, 0].set_ylabel("recon", fontsize=9)
    fig.suptitle(title or f"Denoising at σ={sigma}", fontsize=11)
    return _savefig(fig, filename or f"denoising_sigma{sigma}.png")


# --------------------------------------------------------------------------- #
# Sample grid of raw data
# --------------------------------------------------------------------------- #

def plot_class_samples(class_to_ds: Dict[str, tf.data.Dataset],
                       n_per_class: int = 6,
                       filename: str = "class_samples.png") -> str:
    classes = list(class_to_ds.keys())
    fig, axes = plt.subplots(len(classes), n_per_class,
                             figsize=(1.3 * n_per_class, 1.3 * len(classes)))
    for r, c in enumerate(classes):
        x = _take_batch(class_to_ds[c], n_per_class)
        for i in range(n_per_class):
            ax = axes[r, i] if len(classes) > 1 else axes[i]
            ax.imshow(x[i, :, :, 0], cmap="gray"); ax.axis("off")
        (axes[r, 0] if len(classes) > 1 else axes[0]).set_ylabel(c, fontsize=9)
    fig.suptitle("Medical MNIST samples", fontsize=11)
    fig.tight_layout()
    return _savefig(fig, filename)
