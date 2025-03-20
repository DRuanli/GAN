import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from src.GAN import logger


class GANEvaluation:
    def __init__(self, config):
        self.config = config
        os.makedirs(self.config.evaluation_output_path, exist_ok=True)

    def load_models(self):
        """Load generator and discriminator models"""
        logger.info("Loading GAN models")
        self.generator = tf.keras.models.load_model(self.config.generator_model_path)
        self.discriminator = tf.keras.models.load_model(self.config.discriminator_model_path)

    def load_test_data(self):
        """Load test images for evaluation"""
        logger.info("Loading test data")

        # Find paths containing photos
        photo_dir = self.find_directory_with_images(self.config.test_data_path, "photo")
        monet_dir = self.find_directory_with_images(self.config.test_data_path, "monet")

        if not photo_dir or not monet_dir:
            logger.warning("Could not find photo and Monet directories")
            photo_dir = monet_dir = self.find_any_image_directory(self.config.test_data_path)

        logger.info(f"Found images in {photo_dir} and {monet_dir}")

        # Load test photos
        photo_files = self.get_image_files(photo_dir)
        self.test_photos = self.load_images(
            photo_files[:self.config.num_samples],
            (256, 256)  # Hardcoded for compatibility with models
        )

        # Load real Monet paintings for comparison
        monet_files = self.get_image_files(monet_dir)
        self.real_monet = self.load_images(
            monet_files[:self.config.num_samples],
            (256, 256)  # Hardcoded for compatibility with models
        )

    def find_directory_with_images(self, root_dir, name_contains):
        """Find a directory that contains images and has 'name_contains' in its path"""
        for root, dirs, files in os.walk(root_dir):
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if image_files and name_contains.lower() in root.lower():
                return root
        return None

    def find_any_image_directory(self, root_dir):
        """Find any directory containing images"""
        for root, dirs, files in os.walk(root_dir):
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if image_files:
                return root
        return root_dir

    def get_image_files(self, directory):
        """Get all image files in a directory"""
        extensions = ['.jpg', '.jpeg', '.png']
        files = []
        for ext in extensions:
            files.extend([os.path.join(directory, f) for f in os.listdir(directory)
                          if f.lower().endswith(ext)])
        return files

    def load_images(self, file_paths, image_size):
        """Load and preprocess images from file paths"""
        images = []
        for path in file_paths:
            try:
                img = tf.io.read_file(path)
                img = tf.io.decode_jpeg(img, channels=3)
                img = tf.image.resize(img, image_size)
                img = (tf.cast(img, tf.float32) / 127.5) - 1  # Normalize to [-1, 1]
                images.append(img)
            except Exception as e:
                logger.warning(f"Error loading image {path}: {e}")

        if images:
            return tf.stack(images)
        return None

    def generate_samples(self):
        """Generate Monet-style images from test photos"""
        logger.info("Generating Monet-style samples")
        generated = self.generator(self.test_photos, training=False)

        # Ensure generated images have the correct dimensions for discriminator
        if generated.shape[1:3] != (256, 256):
            logger.info(f"Resizing generated images from {generated.shape[1:3]} to (256, 256)")
            generated = tf.image.resize(generated, (256, 256))

        self.generated_images = generated
        return self.generated_images

    def calculate_metrics(self):
        """Calculate quality metrics for the generated images"""
        metrics = {}

        # Calculate discriminator scores
        try:
            # Ensure both real and generated images have the right dimensions
            real_monet = tf.image.resize(self.real_monet, (256, 256))
            generated_images = tf.image.resize(self.generated_images, (256, 256))

            real_scores = self.discriminator(real_monet, training=False)
            fake_scores = self.discriminator(generated_images, training=False)

            metrics["avg_real_score"] = float(tf.reduce_mean(real_scores))
            metrics["avg_fake_score"] = float(tf.reduce_mean(fake_scores))
            metrics["score_difference"] = metrics["avg_real_score"] - metrics["avg_fake_score"]

            logger.info(f"Real score: {metrics['avg_real_score']:.4f}, Fake score: {metrics['avg_fake_score']:.4f}")
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")

        return metrics

    def save_generated_images(self):
        """Save generated images for visual inspection"""
        logger.info("Saving generated images")

        save_dir = os.path.join(self.config.evaluation_output_path, "generated_samples")
        os.makedirs(save_dir, exist_ok=True)

        # Create a grid of original vs generated images
        plt.figure(figsize=(20, 10))

        for i in range(min(5, len(self.test_photos))):
            # Display original photo
            plt.subplot(2, 5, i + 1)
            plt.title("Original")
            photo = (self.test_photos[i] + 1) * 0.5  # Denormalize
            plt.imshow(photo)
            plt.axis('off')

            # Display generated Monet-style image
            plt.subplot(2, 5, i + 6)
            plt.title("Generated")
            generated = (self.generated_images[i] + 1) * 0.5  # Denormalize
            plt.imshow(generated)
            plt.axis('off')

        comparison_path = os.path.join(save_dir, "comparison.png")
        plt.savefig(comparison_path)
        plt.close()

        # Save individual generated images
        for i, img in enumerate(self.generated_images):
            img = (img + 1) * 127.5  # Convert from [-1, 1] to [0, 255]
            img = tf.cast(img, tf.uint8)
            img_path = os.path.join(save_dir, f"generated_{i}.{self.config.save_format}")

            if self.config.save_format == 'png':
                tf.io.write_file(img_path, tf.io.encode_png(img))
            else:
                tf.io.write_file(img_path, tf.io.encode_jpeg(img))

        return comparison_path, save_dir

    def evaluate(self):
        """Evaluate GAN performance"""
        self.load_models()
        self.load_test_data()
        self.generate_samples()

        # Calculate metrics
        metrics = self.calculate_metrics()

        # Save generated images
        self.comparison_path, self.samples_dir = self.save_generated_images()

        # Save metrics directly using json module
        metrics_path = os.path.join(self.config.evaluation_output_path, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        return metrics

    def log_into_mlflow(self):
        """Log evaluation metrics and artifacts to MLflow with error handling"""
        logger.info("Attempting to log to MLflow")

        if not self.config.mlflow_uri:
            logger.info("No MLflow URI configured, skipping MLflow logging")
            return

        try:
            mlflow.set_registry_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            if tracking_url_type_store not in ["http", "https"]:
                logger.warning(f"MLflow tracking URI scheme '{tracking_url_type_store}' not supported")
                return

            with mlflow.start_run():
                # Log parameters
                mlflow.log_params(self.config.all_params)

                # Log metrics from file
                metrics_path = os.path.join(self.config.evaluation_output_path, "metrics.json")
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                        for key, value in metrics.items():
                            if isinstance(value, (int, float)):
                                mlflow.log_metric(key, value)

                # Log artifacts
                mlflow.log_artifact(self.comparison_path)

                # Log models
                mlflow.keras.log_model(self.generator, "generator_model")
                mlflow.keras.log_model(self.discriminator, "discriminator_model")

        except Exception as e:
            logger.warning(f"Error logging to MLflow: {e}")