# Keras Models (`.keras` format)

This directory contains Keras models saved during the development of the keyword spotting project.

## Contents:

- Models are saved in the Keras v3 native format (a directory structure zipped into a `.keras` file).
- Filenames typically indicate the model architecture, key training parameters/data, and the test accuracy achieved (e.g., `final_model_test_acc_0.87.keras`).

## Generation:

- These models are primarily generated and saved by the main Jupyter Notebook located in the `notebooks/` directory (e.g., `keyword_spotting_development.ipynb`).
- The notebook includes `ModelCheckpoint` callbacks that save the best performing model (based on validation accuracy) during training.
- Additionally, after final evaluation on the test set, the notebook contains logic to manually save the chosen model with a descriptive name if it meets certain performance criteria.

## Note on Reproducibility:

- Due to the inherent stochasticity (randomness) in neural network training (e.g., weight initialization, data shuffling, dropout), running the training notebook multiple times, even with the same settings and random seeds, may result in models with slightly different final weights and performance metrics.
- The models stored here represent specific training runs. The primary model used for generating the final TFLite versions and reported results is typically the one achieving the highest reproducible test accuracy (e.g., `final_model_test_acc_0.87.keras` which was trained on Colab and then evaluated).

## Usage:

These `.keras` models can be:

- Loaded back into Python/Keras for further evaluation, fine-tuning, or inspection.
- Used as the input for conversion to TensorFlow Lite (`.tflite`) format, as demonstrated in the main Jupyter Notebook.
