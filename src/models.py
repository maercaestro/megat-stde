import torch
import torch.nn as nn

#----------------------------------------
#1: Class for MLP1D testing, this will be used for time comparison
#----------------------------------------
class MLP1D(nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.net(x)


#----------------------------------------
#2: Class for MLP testing, this will be used for multiple derivative equations
#----------------------------------------


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_size=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.net(x)
