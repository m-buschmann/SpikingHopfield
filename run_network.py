from model import FullyConnectedLeakyNetwork
from helper import test_image, load_image
from snntorch import spikegen
from skimage.transform import resize
import os


# Define hyperparameters
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
threshold = 0.5

# Load the data
data, train_loader, data_img = load_image(w,h)

# Spiking Data
spike_data = []
for i in range (len(train_loader)):
  data_it,  = next(data)
  spike_data.append(spikegen.rate(data_it, num_steps=num_steps, gain=gain))

 # Create the model
model = FullyConnectedLeakyNetwork(n_neurons, learning_rate, look_back, epochs, iterations, plus, minus, threshold = threshold)

# Initialize membrane potentials
mem = model.leaky.init_leaky()

spike_data_flat = [spike.view(num_steps, -1) for spike in spike_data]

# Train weights using the provided training method
model.train_weights(spike_data_flat, mem, num_steps, learning_rate, look_back, epochs)

outputs = []
image_overlap_list = []

noise_levels = [0.0, 0.2, 0.5]
hyperparameter_title = f'lr_{learning_rate}_epoch_{epochs}_lookback_{look_back}_gain_{gain}'

# Create a directory for the hyperparameters
output_dir = f'output_images/{hyperparameter_title}'
os.makedirs(output_dir, exist_ok=True)

for index, img in enumerate(data_img):
  img = resize(img, (w, h), mode='reflect')
  for noise_level in noise_levels:
    spike_test_data, spike_test_data_noisy, output_spikes, spike_frequency, image_overlap = test_image(model, img, index, num_steps, output_dir, noise_level, w, gain, iterations)
    outputs.append((spike_test_data, spike_test_data_noisy, output_spikes))
    image_overlap_list.append(image_overlap)


