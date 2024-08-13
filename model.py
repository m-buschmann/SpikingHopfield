# -*- coding: utf-8 -*-

#TODO: method to load images
# TODO: write comments + docstrings

import snntorch as snn

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


import matplotlib.pyplot as plt
import numpy as np
import itertools

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import skimage.data
from skimage.color import rgb2gray

from skimage.transform import resize


from skimage import data as skimage_data
from torch.utils.data import DataLoader
from snntorch import spikegen

from snntorch import surrogate

import tensorflow as tf



import torch
import torch.nn as nn
import snntorch as snn

from helper import preprocessing, add_noise

class FullyConnectedLeakyNetwork(nn.Module):
    # was macht beta nochmal?
    def __init__(self, n_neurons, learning_rate, look_back, epochs, iterations, plus, minus, beta=0.9, threshold = 0.5):
        super(FullyConnectedLeakyNetwork, self).__init__()
        self.n_neurons = n_neurons
        self.learning_rate = learning_rate
        self.look_back = look_back
        self.epochs = epochs
        self.iterations = iterations
        self.plus = plus
        self.minus = minus
        self.beta = beta

        # Fully connected weights, no self-connections
        self.weights = nn.Parameter(torch.zeros(n_neurons, n_neurons))
        self.weights.requires_grad = False  # Hebbian learning does not use gradient descent

        # Initialize Leaky neurons
        self.leaky = snn.Leaky(beta=beta , learn_beta = True, threshold = threshold, learn_threshold=True)


    def forward(self, x, mem, n_steps, iterations = 100):
      all_spikes = []
      spk = torch.where(x[0] == 1, 1, 0).float()

      for iteration in range(iterations):
          spike_trace = []
          for step in range(n_steps):
              # Get the input spikes at current time step
              spikes_t = x[step].view(-1).float()

              # Update membrane potential using leaky neurons
              spk, mem = self.leaky(spikes_t + torch.matmul(self.weights, spk), mem)

              # Collect spikes
              spike_trace.append(spk)

          # Update all_spikes with the latest spike trace
          all_spikes.append(spike_trace)

          # Optional: Adjust weights based on the spike trace (this is more advanced and depends on your specific training strategy)
          # self.weights.data = self.hebbian_update(spike_trace, self.weights.data)

      # Return the spikes from the last iteration and the final membrane potentials
      return torch.stack(all_spikes[-1]), mem

    def hebbian_update(self, spike_trace, W, spike_frequency, learning_rate=0.1, lookback=5):
        learning_rate = self.learning_rate
        #lookback = self.lookback
        num_traces = min(len(spike_trace), lookback)
        hebbian_contributions = np.zeros_like(W)
        plus = self.plus
        minus = self.minus

        for i in range(-num_traces, 0):  # Iterate over the last `lookback` entries
            # Compare directly: 1 if both are the same, -1 if different
            update_matrix = np.where(spike_trace[i] == spike_trace[-1], plus, minus)

            # Compute the outer product of spike_trace[i] and spike_trace[-1] for centering
            spk_outer = np.outer(spike_trace[i].detach().numpy(), spike_trace[-1].detach().numpy())

            # Apply centering by subtracting the outer product of spike frequencies
            centered_update = spk_outer - np.outer(spike_frequency, spike_frequency)

            # Combine the update_matrix with centered update
            hebbian_contributions += centered_update * update_matrix

        # Normalize Hebbian contributions
        W += learning_rate * hebbian_contributions / num_traces

        return W

    def train_weights(self, train_data, mem, n_steps, learning_rate, look_back, epochs = 10):
        print("Start to train weights...")
        num_data = len(train_data)  # number of images
        print(f"Number of images: {num_data}")

        W = self.weights.data.numpy()

        for e in range(epochs):
          for img_idx in tqdm(range(num_data)):
              spikes_img = train_data[img_idx]
              spike_counts = spikes_img.sum(dim=0).numpy() # Summe der spikes über alle Zeitschritte (Shape (256))


              spike_frequency = spike_counts / n_steps # Spike Frequenz für jedes Neuron
              spike_trace = []

              for t in range(n_steps):
                  spikes_t = spikes_img[t].view(-1).float()  # Get spikes at time t  für jedes Neuron (sphape (256))


                  # Update membrane potential and spikes
                  spk, mem = self.leaky(spikes_t, mem)

                  spike_trace.append(spk)

                  W = self.hebbian_update(spike_trace, W, spike_frequency, learning_rate, look_back)


              np.fill_diagonal(W, 0)  # Setze diagonale Elemente auf 0, um Selbstverbindungen zu vermeiden
        W /= (num_data * n_steps)  # Normalisiere die Gewichtsmatrix

        print(f"Trained weights (sample): {W[:5, :5]}")



        self.weights.data = torch.tensor(W, dtype=torch.float32)  # Aktualisiere die Gewichte im Model



