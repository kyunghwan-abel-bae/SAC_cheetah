import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPContinuousQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256,), activation_fn=F.relu):
        super(MLPContinuousQNetwork, self).__init__()
        self.input_layer = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        self.activation_fn = activation_fn
    
    def forward(self, s, a):
        x = torch.cat((s, a), dim=1)
        x = self.activation_fn(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fn(hidden_layer(x))
        x = self.output_layer(x)

        return x