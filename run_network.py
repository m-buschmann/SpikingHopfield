from model import FullyConnectedLeakyNetwork
from helper import preprocessing, test_image, calculate_cosine_similarity
import snntorch as snn
from snntorch import spikegen
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from skimage import data as skimage_data
from skimage.color import rgb2gray
from skimage.transform import resize
import tensorflow as tf



num_steps =100
batch_size = 1
w = 16
h = w
n_neurons = w*h
iterations = 100
epochs = 10
last_steps = num_steps-int(num_steps*0.1)
gain = 0.325
learning_rate = 0.035
look_back = 5
plus = 0.1
minus = -0.07
threshold = 0.4

# Load data
camera = skimage_data.camera()
astronaut = rgb2gray(skimage_data.astronaut())
horse = skimage_data.horse()
coffee = rgb2gray(skimage_data.coffee())

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Get a sample image
image1 = 255-train_images[0]
image2 = 255-train_images[1]
image3 = 255-train_images[2]
image4 = 255-train_images[3]

#data = [camera, astronaut, horse, coffee]
data_img = [image1, image2, image3]
data = [preprocessing(d, w,h) for d in data_img]

# Iterate through minibatches
train_loader = DataLoader(data, batch_size=1, shuffle=True)
data = iter(train_loader)

# Spiking Data
spike_data = []
for i in range (len(train_loader)):
  data_it,  = next(data)
  spike_data.append(spikegen.rate(data_it, num_steps=num_steps, gain=gain))

 # Create the model
model = FullyConnectedLeakyNetwork(n_neurons, learning_rate, look_back, epochs, iterations, plus, minus, threshold = threshold)

# Initialize membrane potentials
mem = model.leaky.init_leaky()

# Forward pass through the network
spike_data_flat = []

for i in range (len(spike_data)):
  spike_data_flat.append(spike_data[i].view(num_steps, -1))

# Train weights using the provided training method
model.train_weights(spike_data_flat, mem, num_steps, learning_rate, look_back, epochs)

outputs = []
image_overlap_list = []

noise_levels = [0.0, 0.2, 0.5]

for img in data_img:
  img = resize(img, (w, h), mode='reflect')
  for noise_level in noise_levels:
    spike_test_data, spike_test_data_noisy, output_spikes, spike_frequency, image_overlap = test_image(model, img, num_steps, noise_level)
    outputs.append((spike_test_data, spike_test_data_noisy, output_spikes))
    image_overlap_list.append(image_overlap)


