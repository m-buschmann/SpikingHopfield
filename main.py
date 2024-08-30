from model import SpikingHopfield
from functions import test_image, load_image
from snntorch import spikegen
from skimage.transform import resize
import os

# Run the model with the specified hyperparameters
  
# Define hyperparameters for the simulation
num_steps = 100  # Total time steps for spike generation
batch_size = 1  # Size of data batches for training
w = 16  # Width of the image
h = w  # Height of the image (square)
n_neurons = w * h  # Total number of neurons (corresponds to pixels in the image)
iterations = 100  # Total number of training iterations
epochs = 10  # Number of epochs for training
last_steps = num_steps - int(num_steps * 0.1)  # Steps used for final evaluation
gain = 0.345  # Gain factor for spike generation
learning_rate = 0.035  # Learning rate for training
look_back = 5  # Look back period for input data
plus = 0.1  # Positive threshold for leaky neurons
minus = -0.07  # Negative threshold for leaky neurons
threshold = 0.505  # Threshold for firing neurons

# Load the training data
data, train_loader, data_img = load_image(w,h)

# Collect spiking data from training images
spike_data = []
for i in range (len(train_loader)):
  # Get the next batch of data
  data_it,  = next(data)
  # Generate rate encoded spike data
  spike_data.append(spikegen.rate(data_it, num_steps=num_steps, gain=gain))

# Create the model instance with the specified hyperparameters
model = SpikingHopfield(n_neurons, learning_rate, look_back, epochs, iterations, plus, minus, threshold = threshold)

# Initialize membrane potentials for the model
mem = model.leaky.init_leaky()

# Flatten the spike data for training
spike_data_flat = [spike.view(num_steps, -1) for spike in spike_data]

# Train weights using the provided training method
# the model gets rate encoded images to train on, and will be tested on the same images with different noise levels added
# the goal is to see how well the model can reconstruct the original image from the noisy input
model.train_weights(spike_data_flat, mem, num_steps, learning_rate, look_back, epochs)

# Define noise levels to test
noise_levels = [0.0, 0.2, 0.3, 0.5, 0.75]
# Create a title for the output directory based on hyperparameters
hyperparameter_title = f'lr_{learning_rate}_epoch_{epochs}_lookback_{look_back}_gain_{gain}'

# Create a directory to store output images for the different hyperparameter settings
output_dir = f'output_images/{hyperparameter_title}'
os.makedirs(output_dir, exist_ok=True)

# Evaluate the model on test images
for index, img in enumerate(data_img):
  img = resize(img, (w, h), mode='reflect')  #Resize the image to the model's input size
  for noise_level in noise_levels:
    # Test the image and obtain various results including overlap
    spike_test_data, spike_test_data_noisy, output_spikes, spike_frequency, image_overlap = test_image(model, img, index, num_steps, output_dir, noise_level, w, gain, iterations)




