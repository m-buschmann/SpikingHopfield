import itertools
import os
from model import SpikingHopfield
from functions import test_image, load_image
from snntorch import spikegen
from skimage.transform import resize
import gc

# Do a grid search over hyperparameters to find the best set

# Define constants for the simulation
num_steps = 100  # Total time steps for the simulation
batch_size = 1  # Size of data batches for training
w = 16  # Width of the image
h = w  # Height of the image (square)
n_neurons = w * h  # Total number of neurons (pixels in the image)
iterations = 100  # Total number of training iterations
last_steps = num_steps - int(num_steps * 0.1)  # Steps to consider for evaluation
plus = 0.1  # Positive threshold for leaky neurons
minus = -0.07  # Negative threshold for leaky neurons

# Load the data
data, train_loader, data_img = load_image(w, h)

# Define hyperparameters to search over
learning_rates = [0.029, 0.03, 0.031]  
gains = [0.31, 0.315, 0.32]  
look_backs = [4]  
epochs = [10]
threshold = [0.465, 0.48, 0.495]

# Create a grid of all possible combinations of hyperparameters
hyperparameter_combinations = list(itertools.product(learning_rates, gains, look_backs, epochs, threshold))

# Define CSV filename to save results
csv_filename = 'hyperparameter_results.csv'

# Write header for CSV file if it doesn't already exist
if not os.path.isfile(csv_filename):
    results_header = ["Learning Rate", "Gain", "Look Back", "Epochs", "Threshold", "Average Overlap"]
    with open(csv_filename, 'w') as f:
        f.write(','.join(results_header) + '\n') 

# Loop through each combination of hyperparameters to find the best set
for lr, g, lb, epo, th in hyperparameter_combinations:
    hyperparameter_title = f'lr_{lr}_gain_{g}_lookback_{lb}_epoch_{epo}_threshold_{th}'
    output_dir = f'output_images/{hyperparameter_title}' # Directory for output images
    os.makedirs(output_dir, exist_ok=True)

    print(f"Testing combination: learning_rate={lr}, gain={g}, look_back={lb}, epochs={epo}, threshold={th}")

    # Prepare to collect spike data for each iteration
    spike_data = []

    # Loop over the training data to collect spike data
    for data_it in train_loader:
        spike_data.append(spikegen.rate(data_it, num_steps=num_steps, gain=g))

    # Flatten the spike data for processing
    spike_data_flat = [spike.view(num_steps, -1) for spike in spike_data]

    # Initialize the model with current set of hyperparameters
    model = SpikingHopfield(n_neurons, lr, lb, epo, iterations, plus, minus, threshold=th)

    # Initialize membrane potentials for the model
    mem = model.leaky.init_leaky()

    # Train the model weights using the training method
    model.train_weights(spike_data_flat, mem, num_steps, lr, lb, epo)

    # Initialize average overlap calculation
    avg_overlap = 0
    noise_levels = [0.2, 0.4] # Different noise levels for testing

    # Evaluate the model using test images
    for index, img in enumerate(data_img):
        img = resize(img, (w, h), mode='reflect') # Resize image to match model input
        for noise_level in noise_levels:
            # Test the image and get the overlap value
            _, _, _, _, image_overlap = test_image(model, img, index, num_steps, output_dir, noise_level, w, g, iterations)

            # Sum the overlap for all images and noise levels
            avg_overlap += image_overlap

    # Calculate the average overlap across all test images and noise levels
    avg_overlap /= len(data_img) * len(noise_levels)

    # Prepare the result for CSV writing
    result = {
        "Learning Rate": lr,
        "Gain": g,
        "Look Back": lb,
        "Epochs": epo,
        "Threshold": th,
        "Average Overlap": avg_overlap
    }

    # Append the result to the CSV file
    with open(csv_filename, 'a') as f:
        f.write(','.join([str(value) for value in result.values()]) + '\n')

    # Free up memory
    del spike_data, spike_data_flat, model, mem
    gc.collect() 

print("Results saved to hyperparameter_results.csv")