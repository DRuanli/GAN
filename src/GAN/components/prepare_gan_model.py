import tensorflow as tf
from src.GAN.entity.config_entity import PrepareGANModelConfig
from src.GAN import logger
from pathlib import Path

class PrepareGANModel:
    def __init__(self, config: PrepareGANModelConfig):
        self.config = config

    def build_generator(self):
        """Builds a U-Net style generator for image-to-image translation"""
        logger.info("Building generator model")

        inputs = tf.keras.layers.Input(shape=self.config.params_image_size)

        # Encoder (downsampling)
        enc1 = self._downsample(inputs, self.config.params_generator_filters)
        enc2 = self._downsample(enc1, self.config.params_generator_filters * 2)
        enc3 = self._downsample(enc2, self.config.params_generator_filters * 4)
        enc4 = self._downsample(enc3, self.config.params_generator_filters * 8)

        # Bottleneck
        bottleneck = self._downsample(enc4, self.config.params_generator_filters * 8)

        # Decoder (upsampling with skip connections)
        dec1 = self._upsample(bottleneck, enc4, self.config.params_generator_filters * 8, apply_dropout=True)
        dec2 = self._upsample(dec1, enc3, self.config.params_generator_filters * 4, apply_dropout=True)
        dec3 = self._upsample(dec2, enc2, self.config.params_generator_filters * 2)
        dec4 = self._upsample(dec3, enc1, self.config.params_generator_filters)

        # Output layer
        outputs = tf.keras.layers.Conv2D(3, 4, padding='same', activation='tanh')(dec4)

        generator = tf.keras.Model(inputs=inputs, outputs=outputs)
        generator.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate, beta_1=0.5),
            loss='mae'
        )

        return generator

    def build_discriminator(self):
        """Builds a PatchGAN discriminator"""
        logger.info("Building discriminator model")

        inputs = tf.keras.layers.Input(shape=self.config.params_image_size)

        # Layer 1
        x = tf.keras.layers.Conv2D(
            self.config.params_discriminator_filters, 4, strides=2, padding='same',
            use_bias=self.config.params_use_bias)(inputs)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        # Layer 2
        x = tf.keras.layers.Conv2D(
            self.config.params_discriminator_filters * 2, 4, strides=2, padding='same',
            use_bias=self.config.params_use_bias)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        # Layer 3
        x = tf.keras.layers.Conv2D(
            self.config.params_discriminator_filters * 4, 4, strides=2, padding='same',
            use_bias=self.config.params_use_bias)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        # Layer 4
        x = tf.keras.layers.Conv2D(
            self.config.params_discriminator_filters * 8, 4, strides=1, padding='same',
            use_bias=self.config.params_use_bias)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        # Output layer
        outputs = tf.keras.layers.Conv2D(1, 4, strides=1, padding='same')(x)

        discriminator = tf.keras.Model(inputs=inputs, outputs=outputs)
        discriminator.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate, beta_1=0.5),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        return discriminator

    def _downsample(self, inputs, filters, size=4, strides=2, apply_batchnorm=True):
        """Helper function for downsampling layers in the generator"""
        x = tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same',
                                   use_bias=self.config.params_use_bias)(inputs)

        if apply_batchnorm:
            x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.LeakyReLU(0.2)(x)

        return x

    def _upsample(self, inputs, skip_connection, filters, size=4, strides=2, apply_dropout=False):
        """Helper function for upsampling layers in the generator"""
        x = tf.keras.layers.Conv2DTranspose(filters, size, strides=strides, padding='same',
                                            use_bias=self.config.params_use_bias)(inputs)

        x = tf.keras.layers.BatchNormalization()(x)

        if apply_dropout:
            x = tf.keras.layers.Dropout(self.config.params_dropout_rate)(x)

        x = tf.keras.layers.ReLU()(x)

        # Skip connection (concatenate)
        x = tf.keras.layers.Concatenate()([x, skip_connection])

        return x

    def prepare_gan_model(self):
        """Prepares both generator and discriminator models and saves them"""
        logger.info("Preparing GAN models")

        generator = self.build_generator()
        discriminator = self.build_discriminator()

        # Save models
        self.save_model(self.config.generator_model_path, generator)
        self.save_model(self.config.discriminator_model_path, discriminator)

        logger.info(f"Generator model saved at: {self.config.generator_model_path}")
        logger.info(f"Discriminator model saved at: {self.config.discriminator_model_path}")

        return generator, discriminator

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model) -> None:
        model.save(path)