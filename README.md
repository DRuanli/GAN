# Monet-Style GAN Project

A complete implementation of a Generative Adversarial Network (GAN) that transforms ordinary photographs into Monet-style paintings, including data pipeline, model training, evaluation, and web application integration.

## Overview

This project implements a CycleGAN architecture to learn the mapping between photos and Monet paintings. The system can generate images in the style of Claude Monet from any photograph input, with a modular pipeline approach that follows MLOps best practices.

## Features

- **Data Ingestion**: Automated downloading and extraction of the Monet dataset
- **Model Architecture**: Implementation of state-of-the-art CycleGAN with generator and discriminator networks
- **Training Pipeline**: End-to-end training workflow with optimized hyperparameters
- **Evaluation Metrics**: Quantitative assessment of generated images using discriminator scores
- **Web Application**: Interactive interface for uploading photos and viewing Monet-style transformations
- **MLflow Integration**: Model tracking and artifact logging capabilities

## Project Structure

```
├── artifacts/               # Generated models and data
├── config/                  # Configuration files
│   └── config.yaml          # Project configuration
├── logs/                    # Application logs
├── research/                # Jupyter notebooks for experimentation
├── src/
│   └── GAN/                 # Main package
│       ├── components/      # Core implementations
│       ├── config/          # Configuration management
│       ├── constants/       # Project constants
│       ├── entity/          # Data classes
│       ├── pipeline/        # Process orchestration
│       └── utils/           # Helper functions
├── static/                  # Web application assets
├── templates/               # HTML templates
│   ├── index.html           # Upload interface
│   └── result.html          # Results display
├── main.py                  # Main execution script
├── params.yaml              # Hyperparameters
└── requirements.txt         # Dependencies
```

## Installation

1. Clone this repository
```bash
git clone https://github.com/DRuanli/GAN.git
cd GAN
```

2. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Running the Complete Pipeline

The entire pipeline can be executed with:

```bash
python main.py
```

This will run all stages in sequence:
1. Data ingestion
2. Model preparation
3. Model training
4. Model evaluation
5. Web application deployment

### Running Individual Stages

Each pipeline stage can be run independently:

```bash
# Data ingestion
python src/GAN/pipeline/stage_01_data_ingestion.py

# Model preparation
python src/GAN/pipeline/stage_02_prepare_gan_model.py

# Model training
python src/GAN/pipeline/stage_03_train_gan.py

# Model evaluation
python src/GAN/pipeline/stage_04_evaluate_gan.py

# Web application
python src/GAN/pipeline/stage_05_web_app.py
```

### After finish - Deactivate and Remove environment

Deactivate:

```bash
deactivate
```

Remove environment:

```bash
rm -rf .venv
```

### Using the Web Application

After running the pipeline, the web application is accessible at:
```
http://localhost:8080
```

Upload any photo to see it transformed into Monet's style.

## Configuration

Key parameters can be adjusted in `params.yaml`:

- `IMAGE_SIZE`: Dimensions of input/output images
- `BATCH_SIZE`: Number of images per training batch
- `EPOCHS`: Number of training cycles
- `LEARNING_RATE`: Learning rate for Adam optimizer
- `GENERATOR_FILTERS`: Filter count in generator
- `DISCRIMINATOR_FILTERS`: Filter count in discriminator

## Results

Evaluation metrics are stored in `artifacts/evaluation/metrics.json` after running the pipeline, including:
- `avg_real_score`: Average discriminator score for real Monet paintings
- `avg_fake_score`: Average discriminator score for generated images
- `score_difference`: Gap between real and generated scores (smaller is better)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset provided by Kaggle's "I'm Something of a Painter Myself" competition
- Implementation based on the CycleGAN paper by Jun-Yan Zhu et al.