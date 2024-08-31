# Extending the Hopfield Network: Hebbian Learning in Spiking Neural Networks
## A project for the seminar "Modelling of synaptic plasticity" (SoSe 2024) 

## Overview
This project implements a Spiking Hopfield Network to process and analyze images. The model utilizes rate encoded spike data generated from images, applies noise, and evaluates performance based on the cosine similarity of spike frequency and the original image. The Hopfield network serves as a form of associative memory that can recall patterns based on the input it receives.

## Features
- **Spiking Hopfield Network**: Implements an associative memory model using spiking neurons.
- **Image Preprocessing**: Resizes and normalizes input images for better model performance.
- **Noise Addition**: Introduces noise into images, testing the robustness of the network.
- **Spike Data Generation**: Converts preprocessed images into spike trains suitable for SNNs.
- **Performance Evaluation**: Assesses the model's output against the original image using cosine similarity.
- **Visualization**: Plots spike data and output spike frequency for analysis.

## How to run the code
1. Clone the repository from GitHub:
```bash
git clone https://github.com/m-buschmann/SpikingHopfield.git
```
2. Navigate to the cloned repository directory:
```bash
cd SpikingHopfield
```
3. Create a virtual environment (optional but recommended):
```bash
python3 -m venv venv
```
4. Activate the virtual environment:
```bash
source venv/bin/activate
```
5. Install the required dependencies:
```bash
pip install -r requirements.txt
```
6. Run the main.py script:
```bash
python main.py
```

## File Descriptions
- **model.py**: Contains the definition of the SpikingHopfield class, which implements the Spiking Hopfield Network architecture.
- **helper.py**: Includes helper functions for loading images, preprocessing data, adding noise, and testing images through the SNN.
- **main.py**: The main script where the Spiking Hopfield Network is trained and evaluated using sample images.
- **output_images/**: Directory where output images from the testing phase will be saved.
- **gridsearch.py**: Perform grid search for the best hyperparameter.
- **plot_spiking_image.py**: Plot and save the generated rate encoded spiking image as mp4.

## Usage

1. In **helper.py**, load_image: choose the images you want to train and test on.
2. Run **gridsearch.py** to perform a grid search for the best hyperparameters given your images.
3. Set hyperparameters in **main.py**.
4. Run **main.py** to train your network on the images with the hyperparameters, and test how well the network performed with cosine similarity.

If you just want to see how the network performs with the set images and hyperparameters, only run **main.py**.

To visualize the spiking images:
1. If necessary, set the path to your local ffmpeg installation in **plot_spiking_image.py**:
```bash
plt.rcParams['animation.ffmpeg_path'] = 'path/to/your/ffmpeg'
```
2. Run **plot_spiking_image.py**.

## Acknowledgments

The Hopfield Network concepts are adapted from @takyamamoto (https://github.com/takyamamoto/Hopfield-Network/blob/master/train_mnist.py)

The spiking image generation is adapted from Jason K. Eshraghian (https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_1_spikegen.ipynb)

The project was planned and executed by Malin Braatz, Mathilda Buschmann and Marieke Schmiesing.

## License
Licensed under the GNU GPL License, Version 3.0 (the "LICENSE"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

GPLv3: https://www.gnu.org/licenses/gpl-3.0.en.html
 
