"""Central configuration for the DSAI 490 Assignment 1 project.

All hyperparameters, class definitions, and paths live here so every other
module stays focused on its own concern.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Tuple

# The six anatomical regions in Medical MNIST.
CLASS_NAMES: Tuple[str, ...] = (
    "AbdomenCT",
    "BreastMRI",
    "CXR",
    "ChestCT",
    "Hand",
    "HeadCT",
)

IMG_SIZE: int = 64
NUM_CHANNELS: int = 1
LATENT_DIM: int = 16
SEED: int = 42


@dataclass(frozen=True)
class Paths:
    """Absolute paths used at runtime. Override when running locally."""

    drive_root: str = "/content/drive/MyDrive"
    kaggle_json: str = "/content/drive/MyDrive/kaggle/kaggle.json"
    dataset_zip: str = "/content/drive/MyDrive/medical_mnist.zip"
    dataset_dir: str = "/content/medical_mnist"
    checkpoints: str = "checkpoints"
    figures: str = "figures"


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters shared by AE and VAE."""

    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    kl_warmup_epochs: int = 5
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    shuffle_buffer: int = 4096


PATHS = Paths()
TRAIN = TrainConfig()
