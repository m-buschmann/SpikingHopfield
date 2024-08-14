from skimage.util import random_noise
from skimage.filters import threshold_mean
from torchvision import datasets, transforms
import tensorflow as tf
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
import skimage.data
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
from snntorch import spikegen
from torch.utils.data import DataLoader
import numpy as np
from skimage import data as skimage_data
from skimage.color import rgb2gray
import os

def preprocessing(img, w=16, h=16):

    img = resize(img, (w, h), mode='reflect')
    thresh = threshold_mean(img)
    binary = img > thresh
    shift = 2*(binary*1)-1 # Boolian to int {-1,1}
    # Convert NumPy array to a tensor
    tensor = ToTensor()(shift).float()  # Ensure tensor is of type float
    normalize = transforms.Normalize((0,), (1,))
    tensor = normalize(tensor)

    return tensor

# Add noise to the image
def add_noise(img, noise_level=0.2):
    noisy_img = random_noise(img, mode='s&p', amount=noise_level)
    return noisy_img


def test_image(model, image, image_index, num_steps, noise_level=0.2, w=16, gain=0.2, iterations=100):
    # Ensure the output directory exists
    output_dir = 'output_images'
    os.makedirs(output_dir, exist_ok=True)

    # Preprocess the original image
    h = w
    tensor = preprocessing(image, w, h)

    # Generate spike data from the original image
    spike_test_data = spikegen.rate(tensor, num_steps=num_steps, gain=gain)

    # Add noise to the image
    noisy_image = add_noise(image, noise_level=noise_level)

    # Preprocess the noisy image
    noisy_tensor = preprocessing(noisy_image, w, h)

    # Generate spike data from the noisy image
    spike_test_data_noisy = spikegen.rate(noisy_tensor, num_steps=num_steps, gain=gain)

    # Create a plot for the noisy spike data
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(spike_test_data_noisy.sum(dim=0).view(w, h).detach().cpu().numpy(), cmap='gray')
    plt.title("Noisy Spike Data")

    # Flatten spike data for forward pass
    spike_test_data_flat = spike_test_data_noisy.view(num_steps, -1)

    # Forward pass through the network
    output_spikes, _ = model(spike_test_data_flat, model.leaky.init_leaky(), num_steps, iterations)

    # Calculate the final spike frequency
    spike_frequency = output_spikes.sum(dim=0) / num_steps

    # Create a plot for the output spikes
    plt.subplot(1, 3, 2)
    plt.imshow(spike_frequency.view(w, h).detach().cpu().numpy(), cmap='gray')
    plt.title("Output Spike Frequency")

    # Calculate the overlap with the original non-spiking image
    image_overlap = calculate_cosine_similarity(spike_frequency, image)
    print(f"Overlap with original image (Noise Level {noise_level}): {image_overlap}")

    # Save the plot
    plt.tight_layout()
    plot_filename = os.path.join(output_dir, f'output_image_{image_index}_noise_{noise_level}.png')
    plt.savefig(plot_filename)
    plt.close()  # Close the plot to free up memory

    return spike_test_data, spike_test_data_noisy, output_spikes, spike_frequency, image_overlap

def calculate_cosine_similarity(output_spike_frequency, original_image):
    # Convert original image to a PyTorch tensor if it isn't already
    if not isinstance(original_image, torch.Tensor):
        original_image = torch.tensor(original_image, dtype=torch.float32)

    # Flatten the output spikes and original image
    output_flat = output_spike_frequency.view(-1)
    original_flat = original_image.view(-1).float()

    # Normalize both vectors
    output_flat = output_flat - output_flat.mean()
    output_flat = output_flat / (output_flat.std() + 1e-10)

    original_flat = original_flat - original_flat.mean()
    original_flat = original_flat / (original_flat.std() + 1e-10)

    # Calculate the cosine similarity
    dot_product = torch.dot(output_flat, original_flat)
    norm_output = torch.norm(output_flat)
    norm_original = torch.norm(original_flat)

    if norm_output.item() == 0 or norm_original.item() == 0:
        return 0  # Handle cases where one vector has no spikes at all
    else:
        overlap = dot_product / (norm_output * norm_original)

    return overlap.item()

def load_image(w=16, h=16):
    # Load data
    camera = skimage_data.camera()
    astronaut = rgb2gray(skimage_data.astronaut())
    horse = skimage_data.horse()
    coffee = rgb2gray(skimage_data.coffee())

    mnist = tf.keras.datasets.mnist
    (train_images, _), _ = mnist.load_data()

    # Get a sample image
    image1 = 255-train_images[0]
    image2 = 255-train_images[1]
    image3 = 255-train_images[2]
    image4 = 255-train_images[3]

    #data = [camera, astronaut, horse, coffee]
    data_img = [image1, image3]
    data = [preprocessing(d, w,h) for d in data_img]

    # Iterate through minibatches
    train_loader = DataLoader(data, batch_size=1, shuffle=True)
    data = iter(train_loader)

    return data, train_loader, data_img