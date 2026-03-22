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
    
class HardToOptimizeMNIST(nn.Module):
    def __init__(self):
        super(HardToOptimizeMNIST, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 32)
        self.fc5 = nn.Linear(32, 32)
        self.fc6 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)     # Flatten the image
        x = torch.sigmoid(self.fc1(x))  # Sigmoid instead of ReLU
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        x = self.fc6(x)             # No softmax; let loss handle logits
        return x
