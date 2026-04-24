"""Utility helpers: determinism, noise injection, reconstruction metrics."""

from __future__ import annotations

import os
import random
from typing import Tuple

import numpy as np
import tensorflow as tf

from .config import SEED


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy, and TensorFlow for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def add_gaussian_noise(x: tf.Tensor, sigma: float = 0.2) -> tf.Tensor:
    """Add N(0, sigma^2) noise to a batch of images in [0, 1] and clip."""
    noisy = x + tf.random.normal(tf.shape(x), mean=0.0, stddev=sigma)
    return tf.clip_by_value(noisy, 0.0, 1.0)


def mse_per_image(x: tf.Tensor, x_hat: tf.Tensor) -> tf.Tensor:
    """Per-image MSE, averaged over spatial and channel dims."""
    return tf.reduce_mean(tf.square(x - x_hat), axis=[1, 2, 3])


def mean_mse(x: tf.Tensor, x_hat: tf.Tensor) -> float:
    """Batch-mean MSE as a Python float."""
    return float(tf.reduce_mean(mse_per_image(x, x_hat)).numpy())


def mean_ssim(x: tf.Tensor, x_hat: tf.Tensor) -> float:
    """Batch-mean SSIM, assuming inputs are in [0, 1]."""
    ssim = tf.image.ssim(x, x_hat, max_val=1.0)
    return float(tf.reduce_mean(ssim).numpy())


def evaluate_reconstruction(model, dataset: tf.data.Dataset) -> Tuple[float, float]:
    """Compute mean MSE and SSIM over a dataset. Returns (mse, ssim)."""
    mse_sum, ssim_sum, n = 0.0, 0.0, 0
    for batch in dataset:
        x = batch if not isinstance(batch, (tuple, list)) else batch[0]
        x_hat = model(x, training=False)
        if isinstance(x_hat, (tuple, list)):
            x_hat = x_hat[0]
        batch_size = int(tf.shape(x)[0].numpy())
        mse_sum += mean_mse(x, x_hat) * batch_size
        ssim_sum += mean_ssim(x, x_hat) * batch_size
        n += batch_size
    return mse_sum / max(n, 1), ssim_sum / max(n, 1)
