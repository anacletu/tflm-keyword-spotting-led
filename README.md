# Efficient Keyword Spotting for LED Control with TensorFlow Lite for Microcontrollers - WIP

A project to develop an efficient audio keyword spotting system capable of recognizing simple voice commands (e.g., "on", "off", "one", "two", "three") to control an RGB LED. The model is designed and optimized for deployment on resource-constrained microcontrollers (MCUs) using TensorFlow Lite for Microcontrollers (TFLM).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [Project Overview](#project-overview)
- [Core Objectives](#core-objectives)
- [Dataset](#dataset)
- [Methodology & Pipeline](#methodology--pipeline)
- [Technology Stack](#technology-stack)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
  - [Data Preparation](#data-preparation)
  - [Model Training](#model-training)
  - [Model Conversion & Quantization](#model-conversion--quantization)
  - [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Results & Performance](#results--performance)
- [MCU Deployment Concept](#mcu-deployment-concept)
- [Potential Challenges & Future Work](#potential-challenges--future-work)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Overview

This project tackles the challenge of building a lightweight, accurate keyword spotting (KWS) system. The goal is to classify short audio commands from a predefined vocabulary. The recognized commands are then mapped to control the state (on/off) and color (red, green, blue) of an RGB LED, demonstrating a practical application of embedded machine learning.

**Key Advantages of this Edge AI Approach:**

- **Offline Operation:** Works without an internet connection, ensuring reliability and accessibility.
- **Enhanced Privacy:** Voice data is processed locally on the microcontroller, not sent to cloud servers.
- **Low Latency:** Local processing enables faster response times compared to cloud-dependent systems.

## Core Objectives

1.  **Data Preparation:** Process and prepare the Google Speech Commands dataset for training, focusing on a selected vocabulary.
2.  **Model Development:** Design and train a Convolutional Neural Network (CNN) suitable for audio classification.
3.  **Optimization for MCUs:** Convert the trained model to TensorFlow Lite format and apply post-training quantization (int8) to minimize size and computational cost.
4.  **Performance Analysis:** Evaluate the model's accuracy, size, and the impact of quantization.
5.  **Deployment Feasibility:** Discuss and (if time permits) demonstrate the deployment of the model on a target microcontroller (e.g., Arduino Nano 33 BLE Sense, ESP32).

## Dataset

- **Source:** [Google Speech Commands Dataset v0.02](https://www.tensorflow.org/datasets/catalog/speech_commands) (via TensorFlow Datasets).
- **Selected Vocabulary:**
  - `"one"` (mapped to LED Color: RED)
  - `"two"` (mapped to LED Color: GREEN)
  - `"three"` (mapped to LED Color: BLUE)
  - `"on"` (mapped to LED State: ON)
  - `"off"` (mapped to LED State: OFF)
  - `"_silence_"` (background noise)
  - `"_unknown_"` (other words outside the target vocabulary)
- **Preprocessing:** Audio clips are 1 second long, sampled at 16kHz. Features are extracted as Mel-Frequency Cepstral Coefficients (MFCCs).

## Methodology & Pipeline

1.  **Data Loading & Preprocessing:** Load audio, segment silence, balance classes, extract MFCCs.
2.  **Model Architecture:** A compact CNN designed for audio (e.g., 1D/2D Conv layers, Pooling, Dense layers).
3.  **Training:** Using TensorFlow/Keras, with appropriate callbacks for model saving and learning rate scheduling if needed.
4.  **Evaluation:** Metrics include accuracy, precision, recall, F1-score, and confusion matrix.
5.  **TFLM Conversion:**
    - Convert Keras model to `.tflite` format.
    - Apply int8 post-training quantization using a representative dataset.
    - Analyze model size reduction and performance trade-offs.

## Technology Stack

- **Programming Language:** Python 3.x
- **Core ML Libraries:** TensorFlow, Keras
- **Audio Processing:** Librosa
- **Data Handling:** NumPy
- **Plotting & Visualization:** Matplotlib, Seaborn
- **Dataset Management:** TensorFlow Datasets
- **(Optional for MCU):** Arduino IDE / PlatformIO, C/C++ for TFLM runtime.

## Setup & Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/anacletu/tflm-keyword-spotting-led.git
    cd tflm-keyword-spotting-led
    ```

2.  **Create and activate a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **System requirements:**

- FFmpeg (for audio processing)
- sounddevice (for audio playback)

#### TensorFlow Installation Note:

The requirements.txt is configured by default for Apple Silicon (M1/M2/M3) Macs using tensorflow-macos and tensorflow-metal for GPU acceleration.
If you are on a different system (Windows, Linux, or an Intel Mac):

1. Comment out the lines: tensorflow-macos and tensorflow-metal.
2. Uncomment the line: tensorflow (this provides the standard TensorFlow package, which will use your CPU or an available NVIDIA/AMD GPU if properly configured).

## How to Run

_(This section will be filled with specific commands as the scripts are developed)_

### Data Preparation

```bash
# Example:
# python src/prepare_data.py --data_dir datasets/speech_commands --output_dir datasets/processed
```

### Model Training

```bash
# Example:
# python src/train_model.py --processed_data_dir datasets/processed --model_output_dir models
```

### Model Conversion & Quantization

```bash
# Example:
# python src/convert_to_tflite.py --keras_model_path models/keyword_model.h5 --tflite_output_dir models
```

### Evaluation

```bash
# Example:
# python src/evaluate_model.py --tflite_model_path models/keyword_model_quant.tflite --test_data_dir datasets/processed/test
```

## Project Structure

```
tflm-keyword-spotting-led/
├── data/                     # Raw and processed dataset (or symlinks, if large)
│   ├── raw/                  # Google Speech Commands (downloaded here by TFDS)
│   └── processed/            # MFCCs, labels, etc.
├── notebooks/                # Jupyter notebooks for exploration and experimentation
├── src/                      # Source code for the project
│   ├── prepare_data.py       # Script for data loading and preprocessing
│   ├── model_architecture.py # Defines the CNN model
│   ├── train_model.py        # Script for training the model
│   ├── convert_to_tflite.py  # Script for TFLM conversion and quantization
│   ├── evaluate_model.py     # Script for evaluating model performance
│   └── utils.py              # Utility functions (e.g., audio processing, plotting)
├── models/                   # Saved trained models (Keras .h5, .tflite)
├── mcu_deployment/           # Code for microcontroller deployment (e.g., Arduino sketch)
│   └── keyword_spotting_led/
├── requirements.txt          # Python dependencies
├── LICENSE                   # Project license file
└── README.md                 # This file
```

## Results & Performance

_(This section will be updated with key performance metrics once the model is trained and evaluated)_

- **Baseline Model (Float32):**
  - Accuracy: TBD
  - Size: TBD MB
- **Quantized Model (INT8):**
  - Accuracy: TBD
  - Size: TBD KB
  - Accuracy Drop (if any): TBD %
- **Confusion Matrix:** (Image or textual representation to be added)

## MCU Deployment Concept

The final quantized `.tflite` model is intended for deployment on a microcontroller such as an **Arduino Nano 33 BLE Sense** or an **ESP32**.

**Pipeline on MCU:**

1.  Audio input from an onboard or external microphone.
2.  (On-device) MFCC feature extraction from audio frames.
3.  Inference using the TFLM interpreter with the quantized model.
4.  Post-processing of model output to determine the detected keyword.
5.  Control an RGB LED based on the detected command.

**(Details of actual deployment, if completed, will be added here.)**

## Potential Challenges & Future Work

- **Challenges:**
  - Achieving high accuracy with a very small model size.
  - Robustness to real-world noise.
  - On-device feature extraction efficiency.
  - Class imbalance for `_unknown_` and `_silence_`.
- **Future Work:**
  - Implement on-device MFCC extraction for a fully standalone MCU application.
  - Expand the vocabulary.
  - Investigate more advanced quantization techniques (e.g., quantization-aware training).
  - Improve noise robustness (e.g., data augmentation with noise, noise reduction algorithms).
  - Real-time audio processing pipeline on the MCU.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Google Speech Commands Dataset](https://www.tensorflow.org/datasets/catalog/speech_commands) creators for providing the audio data.
- The TensorFlow and TensorFlow Lite development teams for their incredible open-source machine learning libraries.
