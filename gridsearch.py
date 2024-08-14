import itertools
import os
import pandas as pd
from model import FullyConnectedLeakyNetwork
from helper import preprocessing, test_image, calculate_cosine_similarity, load_image
from snntorch import spikegen
from skimage.transform import resize

num_steps = 100
batch_size = 1
w = 16
h = w
n_neurons = w * h
iterations = 100
last_steps = num_steps - int(num_steps * 0.1)
plus = 0.1
minus = -0.07

# Load the data
data, train_loader, data_img = load_image(w, h)

# Define hyperparameters to search over
learning_rates = [0.03, 0.0325, 0.35, 0.37]  # Example values for learning rates
gains = [0.315, 0.32, 0.325, 0.33, 0.335]  # Example values for gains
look_backs = [4, 5, 6]  # Example values for look back
epochs = [10]
threshold = [0.45, 0.48, 0.5, 0.52]

# Create a grid of all possible combinations
hyperparameter_combinations = list(itertools.product(learning_rates, gains, look_backs, epochs, threshold))

best_params = None
best_overlap = -float('inf')
# Define CSV filename
csv_filename = 'hyperparameter_results.csv'

# Write header for CSV file if it doesn't already exist
if not os.path.isfile(csv_filename):
    results_header = ["Learning Rate", "Gain", "Look Back", "Epochs", "Threshold", "Average Overlap"]
    with open(csv_filename, 'w') as f:
        f.write(','.join(results_header) + '\n')

for lr, g, lb, epo, th in hyperparameter_combinations:

    hyperparameter_title = f'lr_{lr}_gain_{g}_lookback_{lb}_epoch_{epo}_threshold_{th}'
    output_dir = f'output_images/{hyperparameter_title}'
    os.makedirs(output_dir, exist_ok=True)

    print(f"Testing combination: learning_rate={lr}, gain={g}, look_back={lb}, epochs={epo}, threshold={th}")

    # Prepare to collect spike data for each iteration
    spike_data = []

    # Loop over train_loader to collect spike data
    for data_it in train_loader:
        spike_data.append(spikegen.rate(data_it, num_steps=num_steps, gain=g))

    spike_data_flat = [spike.view(num_steps, -1) for spike in spike_data]

    # Initialize the model with current hyperparameters
    model = FullyConnectedLeakyNetwork(n_neurons, lr, lb, epo, iterations, plus, minus, threshold=th)

    # Initialize membrane potentials
    mem = model.leaky.init_leaky()

    # Train weights using the provided training method
    model.train_weights(spike_data_flat, mem, num_steps, lr, lb, epo)

    # Evaluate the model on your test images
    avg_overlap = 0
    noise_levels = [0.1, 0.3]

    for index, img in enumerate(data_img):
        img = resize(img, (w, h), mode='reflect')
        for noise_level in noise_levels:
            _, _, _, _, image_overlap = test_image(model, img, index, num_steps, output_dir, noise_level, w, g, iterations)

            # Sum the overlap for all images and noise levels
            avg_overlap += image_overlap

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

    # Append the result to the CSV file directly
    with open(csv_filename, 'a') as f:
        f.write(','.join([str(value) for value in result.values()]) + '\n')

    # Update best parameters if this combination is better
    if avg_overlap > best_overlap:
        best_overlap = avg_overlap
        best_params = (lr, g, lb, epo)

    print(f"Average Overlap: {avg_overlap:.4f}\n")
    print(f"Best Parameters: learning_rate={best_params[0]}, gain={best_params[1]}, look_back={best_params[2]}")
    print(f"Best Overlap: {best_overlap:.4f}")

print("Results saved to hyperparameter_results.csv")