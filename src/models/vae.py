"""Convolutional Variational Autoencoder.

Uses the reparameterization trick and a Gaussian posterior with a standard
normal prior. Loss is ``recon_loss + beta * kl_loss`` where ``beta`` is
scheduled externally by a callback for KL warm-up.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, layers

from ..config import IMG_SIZE, LATENT_DIM, NUM_CHANNELS


class Sampling(layers.Layer):
    """z = mu + exp(0.5 * log_var) * eps, eps ~ N(0, I)."""

    def call(self, inputs):
        mu, log_var = inputs
        eps = tf.random.normal(tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * eps


def build_vae_encoder(latent_dim: int = LATENT_DIM) -> Model:
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS), name="encoder_input")
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    mu = layers.Dense(latent_dim, name="mu")(x)
    log_var = layers.Dense(latent_dim, name="log_var")(x)
    z = Sampling(name="z")([mu, log_var])
    return Model(inputs, [mu, log_var, z], name="vae_encoder")


def build_vae_decoder(latent_dim: int = LATENT_DIM) -> Model:
    inputs = layers.Input(shape=(latent_dim,), name="decoder_input")
    x = layers.Dense(8 * 8 * 128, activation="relu")(inputs)
    x = layers.Reshape((8, 8, 128))(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    # Logits output; final activation is applied in the loss (BCE from logits).
    logits = layers.Conv2DTranspose(
        NUM_CHANNELS, 3, strides=2, padding="same", activation=None, name="logits"
    )(x)
    return Model(inputs, logits, name="vae_decoder")


class VAE(Model):
    """Convolutional VAE with per-pixel BCE reconstruction and KL divergence."""

    def __init__(self, latent_dim: int = LATENT_DIM, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.encoder = build_vae_encoder(latent_dim)
        self.decoder = build_vae_decoder(latent_dim)
        # Beta is updated externally via a callback during KL warm-up.
        self.beta = tf.Variable(1.0, trainable=False, dtype=tf.float32, name="beta")

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.recon_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.loss_tracker, self.recon_tracker, self.kl_tracker]

    def encode(self, x: tf.Tensor) -> tf.Tensor:
        mu, _, _ = self.encoder(x, training=False)
        return mu

    def decode(self, z: tf.Tensor) -> tf.Tensor:
        logits = self.decoder(z, training=False)
        return tf.sigmoid(logits)

    def sample(self, n: int) -> tf.Tensor:
        z = tf.random.normal((n, self.latent_dim))
        return self.decode(z)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        _, _, z = self.encoder(x, training=training)
        logits = self.decoder(z, training=training)
        return tf.sigmoid(logits)

    def _compute_losses(self, x, training):
        mu, log_var, z = self.encoder(x, training=training)
        logits = self.decoder(z, training=training)
        # Per-pixel BCE summed over the image, then averaged over the batch.
        bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=logits)
        recon_loss = tf.reduce_mean(tf.reduce_sum(bce, axis=[1, 2, 3]))
        kl = -0.5 * tf.reduce_sum(
            1.0 + log_var - tf.square(mu) - tf.exp(log_var), axis=1
        )
        kl_loss = tf.reduce_mean(kl)
        total = recon_loss + self.beta * kl_loss
        return total, recon_loss, kl_loss

    def train_step(self, data):
        x = data if not isinstance(data, (tuple, list)) else data[0]
        with tf.GradientTape() as tape:
            loss, recon, kl = self._compute_losses(x, training=True)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.recon_tracker.update_state(recon)
        self.kl_tracker.update_state(kl)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x = data if not isinstance(data, (tuple, list)) else data[0]
        loss, recon, kl = self._compute_losses(x, training=False)
        self.loss_tracker.update_state(loss)
        self.recon_tracker.update_state(recon)
        self.kl_tracker.update_state(kl)
        return {m.name: m.result() for m in self.metrics}


class KLWarmup(tf.keras.callbacks.Callback):
    """Linearly ramp ``beta`` from 0 to 1 over the first ``warmup_epochs``.

    Prevents posterior collapse on small-image VAEs.
    """

    def __init__(self, warmup_epochs: int):
        super().__init__()
        self.warmup_epochs = max(int(warmup_epochs), 0)

    def on_epoch_begin(self, epoch, logs=None):
        if self.warmup_epochs == 0:
            beta = 1.0
        else:
            beta = min(1.0, epoch / float(self.warmup_epochs))
        self.model.beta.assign(beta)


def make_vae(
    latent_dim: int = LATENT_DIM,
    learning_rate: float = 1e-3,
) -> VAE:
    model = VAE(latent_dim=latent_dim)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))
    # Force weight materialization so ModelCheckpoint has something to save.
    model(tf.zeros((1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS)))
    return model
