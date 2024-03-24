# pyright: basic

from typing import Literal
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pdb import set_trace

class SimpleLayeredNet():
  
    def __init__(self, layer_sizes: list[int], min_weight=-3, max_weight=3, step=1):
        """
        Initialize a neural network with the given layer sizes and random weights.
        
        :param layer_sizes: List of integers representing the number of neurons in each layer.
        :param min_weight: Minimum weight value for random initialization.
        :param max_weight: Maximum weight value for random initialization.
        :param step: Step size for random initialization.
        """
    
        # If any layer size is more than 10, raise an error
        if any(size > 10 for size in layer_sizes):
            raise ValueError("Layer sizes must be less than or equal to 10 for easy visualization and understanding.")
        self.layer_sizes = layer_sizes
        self.weights = self.randomize(min_weight, max_weight, step)
    
    def get_activations(self, input: ndarray):
        """
        Evaluate the firing states of neurons across all layers of a neural network given its configuration.
        
        :param input: Array representing the input to the network, with 1s for active (fired) and 0s otherwise.
        :return: A list of arrays, each representing the firing states (1s and 0s) of neurons in each layer.
        """
    
        activations = [input]
        
        for weights in self.weights:
            input_to_next_layer = np.dot(activations[-1], weights)
            next_layer_activations = np.maximum(0, input_to_next_layer)
            activations.append(next_layer_activations)
    
        return activations
    
    def randomize(self, min_weight: int, max_weight: int, step: int):
        """
        Generate random weights for a neural network with the given layer sizes.
        
        :param min_weight: Minimum weight value for random initialization.
        :param max_weight: Maximum weight value for random initialization.
        :param step: Step size for random initialization.
        """
    
        self.weights: list[ndarray] = []
        
        for i in range(len(self.layer_sizes) - 1):
            weights = np.random.randint(min_weight, max_weight, (self.layer_sizes[i], self.layer_sizes[i + 1]))
            self.weights.append(weights)

        return self.weights
    
    def plot(self, input: ndarray | None = None):
        input = input or self.random_input_with_at_least_one_active_output
        activations = self.get_activations(input)
        num_layers = len(activations)
        num_neurons = max(self.layer_sizes)
        max_weight = np.max([np.max(np.abs(weights)) for weights in self.weights])

        plt.figure(figsize=(num_layers * 3, num_neurons * 1.5))

        for i, activation in enumerate(activations):
            layer_size = self.layer_sizes[i]
            vertical_offset = (num_neurons - layer_size) / 4  # Calculate the offset to center the dots
            
            for j, neuron_activated in enumerate(activation):
                dot_y = (j / 2) + vertical_offset  # Apply the offset to y-coordinate
                dot = Circle((i, dot_y), radius=0.05, color='black' if neuron_activated else 'white')
                plt.gca().add_artist(dot)
                dot.set_edgecolor('black')

        for i, weights in enumerate(self.weights):
            layer_size_source = self.layer_sizes[i]
            layer_size_target = self.layer_sizes[i + 1]
            vertical_offset_source = (num_neurons - layer_size_source) / 4
            vertical_offset_target = (num_neurons - layer_size_target) / 4
            
            for j in range(weights.shape[0]):
                for k in range(weights.shape[1]):
                    weight = weights[j, k]
                    if not weight:
                        continue
                    color = 'red' if weight < 0 else 'green'
                    activation = activations[i][j]
                    alpha = 0.1 if not activation else 1
                    # linewidth = np.abs(weight) * activations[i][j]
                    # let's make two lines: one (lighter) for just the weight, and the other one (colored) for the weight * activation. The latter must be behind the former, of course. We'll do this for activated lines only.
                    weight_linewidth = np.abs(weight)
                    total_linewidth = np.abs(weight) * activation
                    draw_total_line = activation > 0
                    linestyle = 'solid' if activation else 'dashed'
                    # plt.plot([i, i + 1], [j / 2 + vertical_offset_source, k / 2 + vertical_offset_target], color=color, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
                    if draw_total_line:
                        plt.plot([i, i + 1], [j / 2 + vertical_offset_source, k / 2 + vertical_offset_target], color=color, alpha=0.5, linestyle=linestyle, linewidth=total_linewidth)
                    plt.plot([i, i + 1], [j / 2 + vertical_offset_source, k / 2 + vertical_offset_target], color=color, alpha=alpha, linestyle=linestyle, linewidth=weight_linewidth)

        plt.gca().invert_yaxis()
        plt.axis('off')
        plt.show()
    
    @property
    def random_input(self):
        """
        Generate a random input array for the neural network with either 0s or 1s.
        """

        return np.random.randint(0, 2, self.layer_sizes[0])
    
    @property
    def random_input_with_at_least_one_active_output(self):
        """
        Generate a random input array for the neural network with either 0s or 1s, ensuring at least one output neuron is active.
        """
        
        while True:
            input = self.random_input
            activations = self.get_activations(input)
            if np.any(activations[-1]):
                return input
            
    @property
    def paths(self):
        """
        Return the weights of the neural network as a list of paths. Each path is a tuple where the first element is either 1 or -1 (for positive/negative weights, respectively) and the second element is an array of numbers where the integer part represents the layer (0-indexed) and the decimal part represents the neuron (0-indexed).

        Each link in the path represents a 1/-1-weighted connection between two neurons, thus the "sum" of all paths is the neural network. Note that any 0-weighted (i.e. absent) connection between two neurons can be represented as a sum of a 1 and a -1 path, so we can always represent the neural network as a sum of paths even if we run out of links.
        """
        remaining_weights = [weights.copy() for weights in self.weights]
        paths: list[tuple[Literal[1, -1], list[float]]] = []
        for starting_layer in range(len(self.layer_sizes) - 1):
            print(f"{starting_layer=}")
            while np.any(remaining_weights[starting_layer:][0]):
                print(f"{remaining_weights=}")
                # pick the path randomly by starting at a random neuron in the first layer, picking a non-zero weight (its sign will be the path's sign), and following the path until we reach the last layer. Preferably we should pick a neuron with the same sign, then a neuron with no connection, then a neuron with the opposite sign.
                path: list[float] = []
                sign: Literal[1, -1] | None = None
                indices_to_pick_from = np.where(np.any(remaining_weights[starting_layer], axis=1))[0]
                # path.append(np.random.choice(indices_to_pick_from) / 10)
                # Let's pick deterministically from the first possible neuron in that layer
                path.append(indices_to_pick_from[0] / 10)
                for layer in range(starting_layer, len(self.layer_sizes) - 1):
                    current_neuron_index = int(path[-1] * 10) % 10
                    print(f"{layer=}, {current_neuron_index=}")
                    # Pick neurons from the next layer having the same sign as our picked sign (if any, otherwise just any neuron)
                    weights_to_next_layer = remaining_weights[layer]
                    if sign is None:
                        indices_to_pick_from = np.where(weights_to_next_layer[current_neuron_index] != 0)[0]
                        if len(indices_to_pick_from) == 0:
                            indices_to_pick_from = np.arange(weights_to_next_layer.shape[1])
                    else:
                        indices_to_pick_from = np.where(weights_to_next_layer[current_neuron_index] == sign)[0]
                        if len(indices_to_pick_from) == 0:
                            indices_to_pick_from = np.where(weights_to_next_layer[current_neuron_index] == 0)[0]
                        if len(indices_to_pick_from) == 0:
                            indices_to_pick_from = np.arange(weights_to_next_layer.shape[1])
                    # neuron_index = np.random.choice(indices_to_pick_from)
                    # let's pick deterministically from the first possible neuron in that layer
                    neuron_index = indices_to_pick_from[0]
                    print(f"{indices_to_pick_from=}, {neuron_index=}")
                    path.append(layer + 1 + neuron_index / 10)
                    if sign is None:
                        sign = np.sign(weights_to_next_layer[current_neuron_index, neuron_index])
                        if sign == 0:
                            sign = np.random.choice([1, -1])
                    # Subtract 1 or -1 from the weights to indicate that we've used this connection
                    remaining_weights[layer][current_neuron_index, neuron_index] -= sign
                if sign is None:
                    raise ValueError("Sign not set (this should never happen)")
                paths.append((sign, path))
                print(f"{sign=}, {path=}")

        return paths
    

if __name__ == '__main__':
    net = SimpleLayeredNet([5, 3, 3, 2])
    net.plot(net.random_input_with_at_least_one_active_output)