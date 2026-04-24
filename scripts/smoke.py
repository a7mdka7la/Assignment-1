"""Smoke test: exercise the full pipeline on synthetic data.

Run this before a Colab training session to catch bugs that AST parsing
can't: subclassed-model checkpointing, KLWarmup callback wiring, the
train_step / test_step plumbing, and every viz helper.

Usage (from repo root):
    python -m scripts.smoke
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import tensorflow as tf

from src.config import CLASS_NAMES, IMG_SIZE, NUM_CHANNELS, PATHS
from src.models.ae import make_autoencoder
from src.models.vae import KLWarmup, make_vae
from src.utils import set_seed, evaluate_reconstruction
from src import viz


BATCH = 16
N_TRAIN = 64
N_VAL = 32


def _fake_dataset(n: int, batch: int = BATCH) -> tf.data.Dataset:
    x = tf.random.uniform((n, IMG_SIZE, IMG_SIZE, NUM_CHANNELS), seed=0)
    return tf.data.Dataset.from_tensor_slices(x).batch(batch).prefetch(tf.data.AUTOTUNE)


def _fake_labeled_dataset(n_per_class: int = 16, batch: int = BATCH) -> tf.data.Dataset:
    per_class = []
    for label in range(len(CLASS_NAMES)):
        x = tf.random.uniform((n_per_class, IMG_SIZE, IMG_SIZE, NUM_CHANNELS),
                              seed=label, minval=label / 10.0,
                              maxval=label / 10.0 + 1.0)
        y = tf.fill((n_per_class,), label)
        per_class.append(tf.data.Dataset.from_tensor_slices((x, y)))
    ds = per_class[0]
    for d in per_class[1:]:
        ds = ds.concatenate(d)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def test_ae_roundtrip(tmpdir: str) -> None:
    print("[smoke] AE: build, train 1 epoch, save, reload...")
    train_ds = _fake_dataset(N_TRAIN)
    val_ds = _fake_dataset(N_VAL)

    model = make_autoencoder()
    weights_path = os.path.join(tmpdir, "ae_smoke.weights.h5")
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=weights_path,
        save_weights_only=True,
        save_best_only=True,
        monitor="val_loss",
        mode="min",
        verbose=0,
    )
    history = model.fit(train_ds, validation_data=val_ds, epochs=1,
                        callbacks=[ckpt], verbose=0)
    assert "loss" in history.history
    assert "val_loss" in history.history, (
        "val_loss missing from logs — ModelCheckpoint(save_best_only) would silently "
        "never fire."
    )
    assert os.path.exists(weights_path), "ModelCheckpoint did not write weights."

    model2 = make_autoencoder()
    model2.load_weights(weights_path)
    x = next(iter(train_ds))
    np.testing.assert_allclose(
        model(x, training=False).numpy(),
        model2(x, training=False).numpy(),
        rtol=1e-5, atol=1e-6,
    )
    print("[smoke] AE OK.")


def test_vae_roundtrip(tmpdir: str) -> None:
    print("[smoke] VAE: build, train 2 epochs with KL warm-up, save, reload...")
    train_ds = _fake_dataset(N_TRAIN)
    val_ds = _fake_dataset(N_VAL)

    model = make_vae()
    weights_path = os.path.join(tmpdir, "vae_smoke.weights.h5")
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=weights_path,
        save_weights_only=True,
        save_best_only=True,
        monitor="val_loss",
        mode="min",
        verbose=0,
    )
    warmup = KLWarmup(warmup_epochs=2)
    history = model.fit(train_ds, validation_data=val_ds, epochs=2,
                        callbacks=[ckpt, warmup], verbose=0)
    for name in ("loss", "recon_loss", "kl_loss", "val_loss"):
        assert name in history.history, f"Missing metric {name} in VAE history."

    # Beta should have moved during the warm-up (started at 0 on epoch 0).
    assert float(model.beta.numpy()) > 0.0, "KLWarmup did not update model.beta."
    assert os.path.exists(weights_path), "VAE ModelCheckpoint did not write weights."

    model2 = make_vae()
    model2.load_weights(weights_path)
    x = next(iter(train_ds))
    # VAE is stochastic, so compare encoder means instead of full outputs.
    np.testing.assert_allclose(
        model.encode(x).numpy(),
        model2.encode(x).numpy(),
        rtol=1e-5, atol=1e-6,
    )
    print("[smoke] VAE OK.")


def test_viz_helpers(tmpdir: str) -> None:
    print("[smoke] viz: run every plotting helper...")
    # Route figure output to tmp so we don't pollute the repo.
    PATHS_figures_original = PATHS.figures
    viz.PATHS = PATHS.__class__(
        drive_root=PATHS.drive_root,
        kaggle_json=PATHS.kaggle_json,
        dataset_zip=PATHS.dataset_zip,
        dataset_dir=PATHS.dataset_dir,
        checkpoints=PATHS.checkpoints,
        figures=tmpdir,
    )

    ae = make_autoencoder()
    vae = make_vae()

    train_ds = _fake_dataset(N_TRAIN)
    labeled_ds = _fake_labeled_dataset()

    # Build a fake "history-like" object for plot_loss_curves.
    class _Hist:
        def __init__(self, d): self.history = d

    histories = {
        "ae_Hand": _Hist({"loss": [1.0, 0.5], "val_loss": [1.1, 0.6]}),
        "vae_Hand": _Hist({
            "loss": [10.0, 8.0], "val_loss": [11.0, 9.0],
            "recon_loss": [9.0, 7.5], "kl_loss": [1.0, 0.5],
        }),
    }
    viz.plot_loss_curves(histories, filename="loss_smoke.png")
    viz.plot_reconstructions(ae, train_ds, n=4, filename="recon_smoke.png")
    viz.plot_reconstructions_grid(
        {c: ae for c in CLASS_NAMES[:2]},
        {c: train_ds for c in CLASS_NAMES[:2]},
        n=3, filename="recon_grid_smoke.png",
    )
    viz.plot_latent_2d(ae, labeled_ds, method="pca",
                       filename="latent_pca_smoke.png", max_points=32)
    # Skip t-SNE when sample count is too low — sklearn requires perplexity < n.
    viz.plot_vae_samples(vae, n=4, filename="vae_samples_smoke.png")
    viz.plot_latent_traversal(vae, dim=0, steps=4, filename="traversal_smoke.png")
    viz.plot_denoising(ae, train_ds, sigma=0.2, n=4, filename="denoise_smoke.png")
    viz.plot_class_samples(
        {c: train_ds for c in CLASS_NAMES[:3]}, n_per_class=3,
        filename="class_samples_smoke.png",
    )
    print("[smoke] viz OK.")

    viz.PATHS = PATHS.__class__(
        drive_root=PATHS.drive_root,
        kaggle_json=PATHS.kaggle_json,
        dataset_zip=PATHS.dataset_zip,
        dataset_dir=PATHS.dataset_dir,
        checkpoints=PATHS.checkpoints,
        figures=PATHS_figures_original,
    )


def test_metrics() -> None:
    print("[smoke] metrics: evaluate_reconstruction...")
    ae = make_autoencoder()
    ds = _fake_dataset(N_VAL)
    mse, ssim = evaluate_reconstruction(ae, ds)
    assert 0.0 <= mse < 1.0, f"MSE out of bounds: {mse}"
    assert -1.0 <= ssim <= 1.0, f"SSIM out of bounds: {ssim}"
    print(f"[smoke] metrics OK (MSE={mse:.4f}, SSIM={ssim:.4f}).")


def main() -> None:
    set_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        test_ae_roundtrip(tmp)
        test_vae_roundtrip(tmp)
        test_viz_helpers(tmp)
        test_metrics()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
