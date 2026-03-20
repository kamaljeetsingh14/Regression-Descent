import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 64, 64], 
                 activation=F.relu, task="classification"):
        """
      
        Parameters
        ----------
        input_dim : int
            Number of input features.
        output_dim : int
            Number of output features/classes.
        hidden_dims : list of int
            Number of neurons in each hidden layer.
        activation : callable
            Activation function between layers.
        task : str, default="classification"
            Either "classification" (apply softmax at output) or "regression".
        """
        super().__init__()
        self.activation = activation
        self.task = task

        # Build hidden layers dynamically
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            in_dim = h_dim
        self.hidden_layers = nn.ModuleList(layers)

        # Output layer
        self.output_layer = nn.Linear(in_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten input if needed
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        if self.task == "classification":
            return F.softmax(x, dim=1)
        return x  # regression