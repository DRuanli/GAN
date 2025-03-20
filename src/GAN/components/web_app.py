import os
import io
import base64
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import numpy as np
from src.GAN import logger


class GANWebApp:
    def __init__(self, config):
        self.config = config
        self.generator = None

        # Set up correct paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '../../../'))

        # Define template and static folders
        template_folder = os.path.join(project_root, 'templates')
        static_folder = os.path.join(project_root, 'static')

        # Ensure directories exist
        results_dir = os.path.join(static_folder, 'results')
        uploads_dir = os.path.join(static_folder, 'uploads')
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(uploads_dir, exist_ok=True)

        logger.info(f"Template folder: {template_folder}")
        logger.info(f"Static folder: {static_folder}")

        self.app = Flask(__name__,
                         template_folder=template_folder,
                         static_folder=static_folder)

        self.setup_routes()

    def setup_routes(self):
        """Set up Flask routes"""

        @self.app.route('/', methods=['GET'])
        def index():
            return render_template('index.html')

        @self.app.route('/generate', methods=['POST'])
        def generate():
            if 'file' not in request.files:
                return redirect(request.url)

            file = request.files['file']
            if file.filename == '':
                return redirect(request.url)

            # Process the image
            image = Image.open(file.stream)
            preprocessed = self.preprocess_image(image)
            result = self.generate_image(preprocessed)

            # Save the result
            result_filename = f"result_{os.path.splitext(file.filename)[0]}.png"
            result_path = os.path.join(self.app.static_folder, 'results', result_filename)
            result.save(result_path)

            # Convert to base64 for display
            buffered = io.BytesIO()
            result.save(buffered, format="PNG")
            result_base64 = base64.b64encode(buffered.getvalue()).decode()

            # Prepare original image for display
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            original_base64 = base64.b64encode(buffered.getvalue()).decode()

            return render_template('result.html',
                                   original=original_base64,
                                   result=result_base64,
                                   result_file=f'results/{result_filename}')

    def load_model(self):
        """Load the generator model"""
        logger.info("Loading generator model for web app")
        self.generator = tf.keras.models.load_model(self.config.generator_model_path)

    def preprocess_image(self, image):
        """Preprocess uploaded image for the model"""
        img = image.resize((256, 256))
        img_array = np.array(img)
        img_array = (img_array / 127.5) - 1  # Normalize to [-1, 1]
        return np.expand_dims(img_array, 0)

    def generate_image(self, input_image):
        """Generate Monet-style image from input"""
        # Generate the image
        generated = self.generator(input_image, training=False)

        # Convert to PIL Image
        generated = (generated[0] + 1) * 127.5  # Denormalize
        generated = np.clip(generated, 0, 255).astype(np.uint8)
        return Image.fromarray(generated)

    def run(self, debug=True, host='0.0.0.0', port=8080):
        """Run the Flask application"""
        self.load_model()
        logger.info(f"Starting web server on {host}:{port}")
        self.app.run(debug=debug, host=host, port=port)