"""
Neural network architectures for policy training.

This module defines the neural network models used to learn control policies
for PMSM torque-tracking control. Currently implements a Multi-Layer Perceptron
(MLP) architecture using Equinox for efficient JAX-based neural network definitions.

The networks are designed to work with JAX's functional programming paradigm,
enabling JIT compilation and automatic differentiation for training.
"""

import jax
import jax.nn as jnn
import jax.numpy as jnp
import equinox as eqx
import diffrax
from typing import Callable


class MLP(eqx.Module):
    """Multi-layer perceptron for policy learning.

    A fully-connected feedforward neural network with configurable layer sizes,
    Leaky ReLU activations in hidden layers, and tanh output activation.
    
    Attributes:
        layers (list[eqx.nn.Linear]): List of linear layers composing the network.
    
    Args:
        layer_sizes (list[int]): List of layer sizes. First element is input 
            dimension, last element is output dimension.
        key (jax.random.PRNGKey): Random key for parameter initialization.
    """
    layers: list[eqx.nn.Linear]
    
    def __init__(self, layer_sizes: list[int], key):
        """Initialize the MLP with specified architecture.
        
        Parameters
        ----------
        layer_sizes : list[int]
            Sizes of each layer in the network.
        key : jax.random.PRNGKey
            Random number generator key for parameter initialization.
        """
        self.layers = []

        # Iterate over consecutive pairs of layer dimensions
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            key, subkey = jax.random.split(key)
            self.layers.append(
                eqx.nn.Linear(fan_in, fan_out, use_bias=True, key=subkey)
            )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the network.
        
        Parameters
        ----------
        x : jnp.ndarray
            Input to the network.
        
        Returns
        -------
        jnp.ndarray
            Network output, squashed to [-1, 1] via tanh activation.
        """
        # Hidden layers: apply linear transformation followed by Leaky ReLU
        for layer in self.layers[:-1]:
            x = jax.nn.leaky_relu(layer(x))

        # Output layer: linear transformation followed by tanh
        return jnp.tanh(self.layers[-1](x))
