import matplotlib.pyplot as plt
from functions import load_image, plot_spiking_image
from snntorch import spikegen

# Display the spiking image

# plt.rcParams['animation.ffmpeg_path'] = '/home/mathilda/anaconda3/envs/mne/bin/ffmpeg'

# Define hyperparameters for the simulation
num_steps = 100  # Total time steps for spike generation
batch_size = 1  # Size of data batches for training
w = 16  # Width of the image
h = w  # Height of the image (square)
gain = 0.325  # Gain factor for spike generation

# Load the training data
data, train_loader, data_img = load_image(w,h)

# Collect spiking data from training images
spike_data = [] 
for i in range (len(train_loader)):
  # Get the next batch of data
  data_it,  = next(data)
  # Generate spike data using the spiking rate function
  spike_data.append(spikegen.rate(data_it, num_steps=num_steps, gain=gain))

# Select a first sample from the spike data
spike_data_sample = spike_data[0] 
# Remove the batch dimension 
spike_data_sample = spike_data_sample[:, 0]  

# Plot the spiking image
plot_spiking_image(spike_data_sample)
