import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from src.GAN import logger
from src.GAN.entity.config_entity import GANTrainingConfig


class GANTraining:
    def __init__(self, config: GANTrainingConfig):
        self.config = config

    def load_models(self):
        """Load the generator and discriminator models"""
        logger.info("Loading generator and discriminator models")
        self.generator = tf.keras.models.load_model(self.config.generator_model_path)
        self.discriminator = tf.keras.models.load_model(self.config.discriminator_model_path)

        # Optimizers
        self.generator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config.params_learning_rate,
            beta_1=self.config.params_beta1
        )
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config.params_learning_rate,
            beta_1=self.config.params_beta1
        )

    def load_dataset(self):
        """Load and prepare image datasets for GAN training"""
        logger.info("Preparing datasets")

        # Find correct data paths
        data_root = self.config.training_data.parent
        monet_dir = self.find_directory_with_images(data_root, "monet")
        photo_dir = self.find_directory_with_images(data_root, "photo")

        if not monet_dir or not photo_dir:
            raise ValueError(f"Could not find monet and photo directories in {data_root}")

        logger.info(f"Found Monet images in {monet_dir}")
        logger.info(f"Found photo images in {photo_dir}")

        # Load Monet paintings dataset
        monet_files = self.get_image_files(monet_dir)
        self.monet_dataset = self.create_dataset_from_files(
            monet_files,
            self.config.params_batch_size,
            self.config.params_image_size
        )

        # Load photo dataset
        photo_files = self.get_image_files(photo_dir)
        self.photo_dataset = self.create_dataset_from_files(
            photo_files,
            self.config.params_batch_size,
            self.config.params_image_size
        )

        # Create training datasets that cycle indefinitely
        self.monet_dataset = self.monet_dataset.repeat()
        self.photo_dataset = self.photo_dataset.repeat()

    def find_directory_with_images(self, root_dir, name_contains):
        """Find a directory that contains images and has 'name_contains' in its path"""
        for root, dirs, files in os.walk(root_dir):
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if image_files and name_contains.lower() in root.lower():
                return root
        return None

    def get_image_files(self, directory):
        """Get all image files in a directory"""
        extensions = ['.jpg', '.jpeg', '.png']
        files = []
        for ext in extensions:
            files.extend([os.path.join(directory, f) for f in os.listdir(directory)
                          if f.lower().endswith(ext)])
        return files

    def create_dataset_from_files(self, file_paths, batch_size, image_size):
        """Create a dataset from image file paths"""
        dataset = tf.data.Dataset.from_tensor_slices(file_paths)
        dataset = dataset.map(
            lambda x: self.load_and_preprocess_image(x, image_size),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def load_and_preprocess_image(self, file_path, image_size):
        """Load and preprocess an image"""
        img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(img, channels=3)
        # Ensure consistent dimensions
        img = tf.image.resize(img, (256, 256))
        img = (tf.cast(img, tf.float32) / 127.5) - 1  # Normalize to [-1, 1]
        return img

    def discriminator_loss(self, real, generated):
        """Discriminator loss function"""
        real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
            tf.ones_like(real), real)
        generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
            tf.zeros_like(generated), generated)
        return (real_loss + generated_loss) * 0.5

    def generator_loss(self, generated):
        """Generator loss function"""
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)(
            tf.ones_like(generated), generated)

    @tf.function
    def train_step(self, real_photos, real_monet):
        """Single training step for image-to-image GAN"""
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate Monet-style image from photo
            fake_monet = self.generator(real_photos, training=True)

            # Ensure consistent dimensions
            if fake_monet.shape[1:3] != (256, 256):
                fake_monet = tf.image.resize(fake_monet, (256, 256))

            # Discriminator predictions
            real_monet_output = self.discriminator(real_monet, training=True)
            fake_monet_output = self.discriminator(fake_monet, training=True)

            # Calculate losses
            gen_loss = self.generator_loss(fake_monet_output)
            disc_loss = self.discriminator_loss(real_monet_output, fake_monet_output)

        # Calculate gradients
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        # Apply gradients
        self.generator_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        return gen_loss, disc_loss

    def save_sample_images(self, epoch, photos):
        """Generate and save sample images"""
        predictions = self.generator(photos, training=False)

        plt.figure(figsize=(12, 12))

        for i in range(min(4, len(predictions))):
            # Display original photo
            plt.subplot(4, 2, i * 2 + 1)
            plt.title("Original")
            photo = (photos[i] + 1) * 0.5  # Denormalize
            plt.imshow(photo)
            plt.axis('off')

            # Display generated Monet-style image
            plt.subplot(4, 2, i * 2 + 2)
            plt.title("Generated")
            prediction = (predictions[i] + 1) * 0.5  # Denormalize
            plt.imshow(prediction)
            plt.axis('off')

        plt.savefig(os.path.join(self.config.root_dir, f'epoch_{epoch}.png'))
        plt.close()

    def train(self):
        """Train the GAN"""
        logger.info(f"Starting GAN training for {self.config.params_epochs} epochs")

        # Get sample photos for visualization
        sample_photos = next(iter(self.photo_dataset.take(1)))

        # Set a reasonable number of steps per epoch
        steps_per_epoch = 100

        # Create iterators
        monet_iter = iter(self.monet_dataset)
        photo_iter = iter(self.photo_dataset)

        for epoch in range(self.config.params_epochs):
            start = time.time()
            logger.info(f"Epoch {epoch + 1}/{self.config.params_epochs}")

            # Training loop
            for step in range(steps_per_epoch):
                real_monet = next(monet_iter)
                real_photos = next(photo_iter)

                gen_loss, disc_loss = self.train_step(real_photos, real_monet)

                if step % 10 == 0:
                    logger.info(f"Step {step}/{steps_per_epoch} - Gen loss: {gen_loss:.4f}, Disc loss: {disc_loss:.4f}")

            # Generate sample images
            self.save_sample_images(epoch + 1, sample_photos)

            # Save models periodically
            if (epoch + 1) % self.config.params_save_interval == 0:
                self.save_models(epoch + 1)

            logger.info(f"Time for epoch {epoch + 1}: {time.time() - start:.2f} sec")

        # Save final models
        self.save_models("final")
        logger.info("GAN training completed")

    def save_models(self, epoch):
        """Save the generator and discriminator models"""
        generator_path = os.path.join(self.config.root_dir, f"generator_epoch_{epoch}.h5")
        discriminator_path = os.path.join(self.config.root_dir, f"discriminator_epoch_{epoch}.h5")

        self.generator.save(generator_path)
        self.discriminator.save(discriminator_path)

        # Save final models with the original names
        if epoch == "final":
            self.generator.save(self.config.trained_generator_path)
            self.discriminator.save(self.config.trained_discriminator_path)
            logger.info(
                f"Final models saved at {self.config.trained_generator_path} and {self.config.trained_discriminator_path}")