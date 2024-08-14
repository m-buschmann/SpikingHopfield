# Spiking Hopfield Network for Image Processing

## Overview
This project implements a Spiking Hopfield Network to process and analyze images. The model utilizes spike data generated from images, applies noise, and evaluates performance based on the cosine similarity of spike frequency and the original image. The Hopfield network serves as a form of associative memory that can recall patterns based on the input it receives.

## Features
- **Spiking Hopfield Network**: Implements an associative memory model using spiking neurons.
- **Image Preprocessing**: Resizes and normalizes input images for better model performance.
- **Noise Addition**: Introduces noise into images, testing the robustness of the network.
- **Spike Data Generation**: Converts preprocessed images into spike trains suitable for SNNs.
- **Performance Evaluation**: Assesses the model's output against the original image using cosine similarity.
- **Visualization**: Plots spike data and output spike frequency for analysis.

## Requirements
This project depends on several Python libraries. To install the required dependencies, you can use the following command:

```bash
pip install -r requirements.txt
```

## File Descriptions
- **model.py**: Contains the definition of the FullyConnectedLeakyNetwork class, which implements the Spiking Hopfield Network architecture.
- **helper.py**: Includes helper functions for loading images, preprocessing data, adding noise, and testing images through the SNN.
- **main.py**: The main script where the Spiking Hopfield Network is trained and evaluated using sample images.
- **output_images/**: Directory where output images from the testing phase will be saved.
- **gridsearch.py**:

