from model import FullyConnectedLeakyNetwork
from helper import preprocessing, test_image, calculate_cosine_similarity, load_image
from snntorch import spikegen
from skimage.transform import resize
import itertools

num_steps =100
batch_size = 1
w = 16
h = w
n_neurons = w*h
iterations = 100
last_steps = num_steps-int(num_steps*0.1)
plus = 0.1
minus = -0.07


data, train_loader, data_img = load_image(w,h)

# Define hyperparameters to search over
learning_rates = [0.03, 0.0325, 0.35, 0.37]  # Example values for learning rates
gains = [0.315, 0.32, 0.325, 0.33, 0.335]             # Example values for gains
look_backs = [4,5,6]              # Example values for look back
epochs = [10]
threshold = [0.45, 0.48, 0.5, 0.52]

# Create a grid of all possible combinations
hyperparameter_combinations = list(itertools.product(learning_rates, gains, look_backs, epochs, threshold))

best_params = None
best_overlap = -float('inf')

for lr, g, lb, epo, th in hyperparameter_combinations:
    print(f"Testing combination: learning_rate={lr}, gain={g}, look_back={lb}, echos={epo}, threshold={th}")
    # Spiking Data
    spike_data = []

    for i in range (len(train_loader)):
        data_it,  = next(data)
        spike_data.append(spikegen.rate(data_it, num_steps=num_steps, gain=g))

    spike_data_flat = []

    for i in range (len(spike_data)):
        spike_data_flat.append(spike_data[i].view(num_steps, -1))


    # Initialize the model with current hyperparameters
    model = FullyConnectedLeakyNetwork(n_neurons, lr, lb, epo, iterations, plus, minus, threshold = th)

    # Initialize membrane potentials
    mem = model.leaky.init_leaky()

    # Train weights using the provided training method
    model.train_weights(spike_data_flat, mem, num_steps, lr, lb, epo)

    # Evaluate the model on your test images
    avg_overlap = 0
    noise_levels = [0.1, 0.3]
    for img in data_img:
        img = resize(img, (w, h), mode='reflect')
        for noise_level in noise_levels:
          _, _, output_spikes = test_image(model, img, num_steps, noise_level=noise_level)

          # Flatten and compute overlap for evaluation
          spike_frequency = output_spikes.sum(dim=0) / num_steps
          overlap = calculate_cosine_similarity(preprocessing(img, w, h), spike_frequency.view(w, h).detach().cpu().numpy())
          avg_overlap += overlap

    avg_overlap /= len(data_img) * len(noise_levels)

    # Update best parameters if this combination is better
    if avg_overlap > best_overlap:
        best_overlap = avg_overlap
        best_params = (lr, g, lb, epo)

    print(f"Average Overlap: {avg_overlap:.4f}\n")

    print(f"Best Parameters: learning_rate={best_params[0]}, gain={best_params[1]}, look_back={best_params[2]}")
    print(f"Best Overlap: {best_overlap:.4f}")
