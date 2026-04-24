"""Convolutional Autoencoder for 64x64 grayscale medical images."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, layers

from ..config import IMG_SIZE, LATENT_DIM, NUM_CHANNELS


def build_encoder(latent_dim: int = LATENT_DIM) -> Model:
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS), name="encoder_input")
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    z = layers.Dense(latent_dim, name="latent")(x)
    return Model(inputs, z, name="ae_encoder")


def build_decoder(latent_dim: int = LATENT_DIM) -> Model:
    inputs = layers.Input(shape=(latent_dim,), name="decoder_input")
    x = layers.Dense(8 * 8 * 128, activation="relu")(inputs)
    x = layers.Reshape((8, 8, 128))(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    outputs = layers.Conv2DTranspose(
        NUM_CHANNELS, 3, strides=2, padding="same", activation="sigmoid", name="reconstruction"
    )(x)
    return Model(inputs, outputs, name="ae_decoder")


class AutoEncoder(Model):
    """Deterministic convolutional autoencoder trained with MSE."""

    def __init__(self, latent_dim: int = LATENT_DIM, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.encoder = build_encoder(latent_dim)
        self.decoder = build_decoder(latent_dim)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def encode(self, x: tf.Tensor) -> tf.Tensor:
        return self.encoder(x, training=False)

    def decode(self, z: tf.Tensor) -> tf.Tensor:
        return self.decoder(z, training=False)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        z = self.encoder(x, training=training)
        return self.decoder(z, training=training)

    def _recon_loss(self, x: tf.Tensor, x_hat: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(tf.square(x - x_hat))

    def train_step(self, data):
        x = data if not isinstance(data, (tuple, list)) else data[0]
        with tf.GradientTape() as tape:
            x_hat = self(x, training=True)
            loss = self._recon_loss(x, x_hat)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        x = data if not isinstance(data, (tuple, list)) else data[0]
        x_hat = self(x, training=False)
        loss = self._recon_loss(x, x_hat)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


def make_autoencoder(
    latent_dim: int = LATENT_DIM,
    learning_rate: float = 1e-3,
) -> AutoEncoder:
    model = AutoEncoder(latent_dim=latent_dim)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))
    # Subclassed models don't always materialize weights from build(input_shape);
    # a throwaway forward pass guarantees variables exist before checkpointing.
    model(tf.zeros((1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS)))
    return model
