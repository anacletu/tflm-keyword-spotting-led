# Data Directory

This directory is used to store the Speech Commands dataset, but **the actual data files are not included in the Git repository** due to their large size.

## Directory Structure

```
data/
├── raw/
│   └── tensorflow_datasets/     # Auto-populated by TensorFlow Datasets
├── processed/                   # For any preprocessed data (if needed)
└── README.md                    # This file
```

## Downloading the Dataset

The Jupyter notebooks in this project are configured to automatically download and prepare the Speech Commands dataset to the correct location. When you run the data exploration notebook, it will:

1. Create necessary subdirectories
2. Download the dataset if not already present
3. Prepare and load the data into TensorFlow Dataset objects

The first time you run the notebook, it may take several minutes to download and prepare the dataset. Make sure you have the system requirements described on the main [README file](../README.md).
