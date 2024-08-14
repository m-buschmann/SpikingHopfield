# -*- coding: utf-8 -*-

# TODO: write comments + docstrings
# TODO: add nice movie plot of noisy images
# TODO: README

import snntorch as snn
import torch
import torch.nn as nn
import numpy as np
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import snntorch as snn

class FullyConnectedLeakyNetwork(nn.Module):
    """
    A fully connected leaky spiking neural network using Hebbian learning rules.

    Attributes:
        n_neurons (int): Number of neurons in the network.
        learning_rate (float): Learning rate for weight updates.
        look_back (int): Number of recent spike traces to consider for Hebbian updates.
        epochs (int): Number of epochs for training.
        iterations (int): Number of iterations for the forward pass.
        plus (float): Positive contribution for Hebbian updates.
        minus (float): Negative contribution for Hebbian updates.
        beta (float): Leak rate for leaky neuron model.
        threshold (float): Threshold for spiking activation.
        weights (torch.nn.Parameter): The weight matrix of the network.
        leaky (snn.Leaky): Leaky neuron model.
    """

    def __init__(self, n_neurons, learning_rate, look_back, epochs, iterations, plus, minus, beta=0.9, threshold = 0.5):
        """
        Initializes the FullyConnectedLeakyNetwork with specified parameters.

        Parameters:
            n_neurons (int): Number of neurons in the network.
            learning_rate (float): Learning rate for weight updates.
            look_back (int): Number of recent spike traces to consider for updates.
            epochs (int): Number of epochs for training.
            iterations (int): Number of iterations for the forward pass.
            plus (float): Positive contribution for learning (Hebbian).
            minus (float): Negative contribution for learning (Hebbian).
            beta (float, optional): Leak rate for leaky neurons. Default is 0.9.
            threshold (float, optional): Spiking threshold for leaky neurons. Default is 0.5.
        """
        super(FullyConnectedLeakyNetwork, self).__init__()
        self.n_neurons = n_neurons
        self.learning_rate = learning_rate
        self.look_back = look_back
        self.epochs = epochs
        self.iterations = iterations
        self.plus = plus
        self.minus = minus
        self.beta = beta

        # Initialize fully connected weights with no self-connections (self-weight is 0)
        self.weights = nn.Parameter(torch.zeros(n_neurons, n_neurons))
        self.weights.requires_grad = False  # Hebbian learning does not use gradient descent

        # Create leaky neurons with learnable parameters for beta and threshold
        self.leaky = snn.Leaky(beta=beta , learn_beta = True, threshold = threshold, learn_threshold=True)


    def forward(self, x, mem, n_steps, iterations = 100):
        """
        Perform forward propagation through the network.

        Parameters:
            x (torch.Tensor): Input spike tensor with shape (n_steps, n_neurons). 
            mem (torch.Tensor): Membrane potential for neurons.
            n_steps (int): Number of time steps for the forward pass.
            iterations (int, optional): Number of iterations to run through the input. Default is 100.

        Returns:
            tuple: A tuple containing:
                - A tensor of spikes from the last time step.
                - The final membrane potentials.
        """
        all_spikes = []

        # Initialize spikes with the first image's spikes
        spk = torch.where(x[0] == 1, 1, 0).float()

        for iteration in range(iterations):
            spike_trace = []
            for step in range(n_steps):
                # Get the input spikes at the current time step
                extern_spikes_t = x[step].view(-1).float()

                # Update membrane potential using leaky neurons
                spk, mem = self.leaky(extern_spikes_t + torch.matmul(self.weights, spk), mem)

                # Collect spikes for each step
                spike_trace.append(spk)

            # Update all_spikes with the latest spike trace from the current iteration
            all_spikes.append(spike_trace)

        # Return the spikes from the last iteration and the final membrane potentials
        return torch.stack(all_spikes[-1]), mem

    def hebbian_update(self, spike_trace, W, spike_frequency, learning_rate=0.1, lookback=5):
        """
        Update weights based on Hebbian learning rules.

        Parameters:
            spike_trace (list): List of spike tensors collected during the forward pass in training.
            W (numpy.ndarray): Current weight matrix of the network.
            spike_frequency (numpy.ndarray): Frequency of spikes for each neuron.
            learning_rate (float, optional): Learning rate for updating weights. Default is 0.1.
            lookback (int, optional): Number of recent spike traces to use for updates. Default is 5.

        Returns:
            numpy.ndarray: Updated weight matrix after applying the Hebbian learning rule.
        """
        learning_rate = self.learning_rate
        num_traces = min(len(spike_trace), lookback)
        hebbian_contributions = np.zeros_like(W) # Initialize contributions to zero
        plus = self.plus
        minus = self.minus

        # Iterate over the last `lookback` entries of the spike trace
        for i in range(-num_traces, 0): 
            # Create an update matrix where spikes match the last trace gain a positive contribution; otherwise, a negative one
            update_matrix = np.where(spike_trace[i] == spike_trace[-1], plus, minus)

            # Compute the outer product of spike_trace[i] and spike_trace[-1]
            spk_outer = np.outer(spike_trace[i].detach().numpy(), spike_trace[-1].detach().numpy())

            # Apply centering by subtracting the outer product of spike frequencies
            centered_update = spk_outer - np.outer(spike_frequency, spike_frequency)

            # Combine the update matrix with the centered update contributions
            hebbian_contributions += centered_update * update_matrix

        # Normalize Hebbian contributions into weight updates
        W += learning_rate * hebbian_contributions / num_traces

        return W

    def train_weights(self, train_data, mem, n_steps, learning_rate, look_back, epochs = 10):
        """
        Train the weights of the network using spike data and Hebbian learning.

        Parameters:
            train_data (list): List of spike trains for training the network.
            mem (torch.Tensor): Initial membrane potentials of the neurons.
            n_steps (int): Number of time steps for the forward pass.
            learning_rate (float): Learning rate for Hebbian weight updates.
            look_back (int): Number of recent spike traces to consider for weight updates.
            epochs (int, optional): Number of training epochs. Default is 10.
        """
        print("Start to train weights...")
        num_data = len(train_data)  # Number of training samples (images)
        print(f"Number of images: {num_data}")

        W = self.weights.data.numpy() # Get current weights in numpy format

        for e in range(epochs):
            for img_idx in tqdm(range(num_data)):
                spikes_img = train_data[img_idx] # Get the spike train for the current image
                spike_counts = spikes_img.sum(dim=0).numpy() # Sum spikes over all time steps

                spike_frequency = spike_counts / n_steps # Calculate spike frequency for each neuron
                spike_trace = []

                for t in range(n_steps):
                    spikes_t = spikes_img[t].view(-1).float()  # Get spikes at time t for each neuron


                    # Update membrane potential and spikes
                    spk, mem = self.leaky(spikes_t, mem)
                    spike_trace.append(spk) # Store the spike data for this time step
                    
                    # Update weights based on the current spike trace
                    W = self.hebbian_update(spike_trace, W, spike_frequency, learning_rate, look_back)

            
                # Set diagonal elements to zero to avoid self-connections
                np.fill_diagonal(W, 0)  
        
        # Normalize the weight matrix after training
        W /= (num_data * n_steps)

        print(f"Trained weights (sample): {W[:5, :5]}")

        # Update the model's weight parameter with the new weights
        self.weights.data = torch.tensor(W, dtype=torch.float32)  



