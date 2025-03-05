import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, n_classes, n_embed):
        super().__init__()
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(n_embed, n_classes)

    def forward(self, x):
        x = self.fc1(self.nonlinear(torch.flatten(x, start_dim = -2)))
        return x
