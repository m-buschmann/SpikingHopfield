from skimage.util import random_noise
from skimage.filters import threshold_mean
from torchvision import transforms
import tensorflow as tf
import torch
from torchvision.transforms import ToTensor
from skimage.transform import resize
import matplotlib.pyplot as plt
from snntorch import spikegen
from torch.utils.data import DataLoader
from skimage import data as skimage_data
from skimage.color import rgb2gray
import os
import snntorch.spikeplot as splt
from IPython.display import HTML



def preprocessing(img, w=16, h=16):
    """
    Preprocess the input image by resizing and converting it into a tensor.
    The image is thresholded to create a binary representation.

    Args:
        img (numpy.ndarray): The input image to preprocess.
        w (int): The width to resize the image to. Defualt is 16.
        h (int): The height to resize the image to. Default is 16.

    Returns:
        torch.Tensor: The preprocessed image as a tensor.
    """
    img = resize(img, (w, h), mode='reflect') # Resize image
    thresh = threshold_mean(img) # Calculate threshold
    binary = img > thresh # Create binary image based on threshold
    shift = 2*(binary*1)-1 # Boolian to int {-1,1}
    tensor = ToTensor()(shift).float()  # Convert NumPy array to a tensor of type float
    normalize = transforms.Normalize((0,), (1,))
    tensor = normalize(tensor) # Normalize tensor

    return tensor

# Add noise to the image
def add_noise(img, noise_level=0.2):
    """
    Add random noise to the input image.

    Args:
        img (numpy.ndarray): The input image to which noise will be added.
        noise_level (float): The amount of noise to add (between 0 and 1).

    Returns:
        numpy.ndarray: The noisy image.
    """
    noisy_img = random_noise(img, mode='s&p', amount=noise_level) # Add salt and pepper noise
    return noisy_img


def test_image(model, image, image_index, num_steps, output_dir, noise_level=0.2, w=16, gain=0.2, iterations=100):
    """
    Test the model with a given image, adding noise and generating spikes. 
    The results are plotted and saved.

    Args:
        model: The model used for testing.
        image (numpy.ndarray): The image input for testing.
        image_index (int): The index of the image for output naming.
        num_steps (int): The number of time steps to simulate.
        output_dir (str): Directory where output images will be saved.
        noise_level (float): The amount of noise to add to the image. Default is 0.2.
        w (int): The width of the image. Default is 16.
        gain (float): Gain factor for spike generation. Default is 0.2.
        iterations (int): Number of iterations for the model. Default is 100.

    Returns:
        tuple: A tuple containing spike test data, noisy spike test data,
               output spikes, spike frequency, and overlap with the original image.
    """
    h = w  # Height is equal to width for square images
    tensor = preprocessing(image, w, h) # Preprocess the original image

    # Generate spike data from the original image
    spike_test_data = spikegen.rate(tensor, num_steps=num_steps, gain=gain)

    # Add noise to the image and preprocess it
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

    # Flatten spike data for network forward pass
    spike_test_data_flat = spike_test_data_noisy.view(num_steps, -1)

    # Perform a forward pass through the model
    output_spikes, _ = model(spike_test_data_flat, model.leaky.init_leaky(), num_steps, iterations)

    # Calculate the final spike frequency for the output
    spike_frequency = output_spikes.sum(dim=0) / num_steps

    # Generate a plot for output spikes
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
    plt.close() 

    return spike_test_data, spike_test_data_noisy, output_spikes, spike_frequency, image_overlap

def calculate_cosine_similarity(output_spike_frequency, original_image):
    """
    Calculate the cosine similarity between the model's output spike frequency and the original image.

    Args:
        output_spike_frequency (torch.Tensor): The output spike frequency from the model.
        original_image (torch.Tensor or numpy.ndarray): The original image to compare.

    Returns:
        float: The cosine similarity (overlap) between the two images.
    """
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

    if norm_output.item() == 0 or norm_original.item() == 0: # Handle edge cases
        return 0  
    else:
        overlap = dot_product / (norm_output * norm_original)

    return overlap.item()

def load_image(w=16, h=16):
    """
    Load and preprocess sample images for use in the model.

    Args:
        w (int): The width to resize images to.
        h (int): The height to resize images to.

    Returns:
        tuple: Contains an iterator over the training data,
               the DataLoader, and the original images used.
    """
    # Load data from skimage sample images
    camera = skimage_data.camera() # Grayscale image
    astronaut = rgb2gray(skimage_data.astronaut())
    horse = skimage_data.horse()
    coffee = rgb2gray(skimage_data.coffee())

    # Load MNIST dataset
    mnist = tf.keras.datasets.mnist
    (train_images, _), _ = mnist.load_data()

    # Obtain sample images from the MNIST dataset
    image1 = 255-train_images[0] # Invert colors
    image2 = 255-train_images[1]
    image3 = 255-train_images[2]
    image4 = 255-train_images[3]

    # Select images for processing
    #data = [camera, astronaut, horse, coffee]
    data_img = [image1, image2, image3]
    data = [preprocessing(d, w,h) for d in data_img] # Preprocess selected images

    # Create a DataLoader for mini-batches
    train_loader = DataLoader(data, batch_size=1, shuffle=True) # Load data in batches of 1
    data = iter(train_loader) # Create an iterator from DataLoader

    return data, train_loader, data_img

def plot_spiking_image(spike_data):
    """
    Plot a spiking image using the provided spike data.
    
    Args:
        spike_data (torch.Tensor): The spike data to plot.
    """
    fig, ax = plt.subplots()
    anim = splt.animator(spike_data, fig, ax)

    HTML(anim.to_html5_video())
    anim.save('animation.mp4')