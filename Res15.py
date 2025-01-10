"""
@property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
        }
'''''
import torch
from torch import nn
from torch.nn import functional as F

class Res15(nn.Module):
    def __init__(self, n_maps):
        """
        Args:
            n_maps (int): Number of feature maps (channels) for the convolutional layers.
        """
        super(Res15, self).__init__()
        self.n_maps = n_maps
        self.conv0 = nn.Conv2d(1, n_maps, (3, 3), padding=(1, 1), bias=False)
        self.n_layers = n_layers = 13
        self.dilation = True

        # Create convolutional layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(n_layers):
            dilation_rate = int(2 ** (i // 3)) if self.dilation else 1
            padding = dilation_rate if self.dilation else 1
            self.convs.append(nn.Conv2d(n_maps, n_maps, (3, 3), padding=padding, dilation=dilation_rate, bias=False))
            self.bns.append(nn.BatchNorm2d(n_maps, affine=False))

    def forward(self, audio_signal, length=None):
        """
        Forward pass for the Res15 model.

        Args:
            audio_signal (Tensor): Input tensor of shape (batch_size, time_steps, features).
            length (Tensor, optional): Tensor representing lengths of input sequences.

        Returns:
            Tensor: Output embeddings of shape (batch_size, n_maps, 1).
            Tensor: Lengths tensor (unchanged from input).
        """
        # Add channel dimension
        x = audio_signal.unsqueeze(1)

        old_x = None
        for i in range(self.n_layers + 1):
            if i == 0:
                y = F.relu(self.conv0(x))
                if hasattr(self, "pool"):
                    y = self.pool(y)
                old_x = y
            else:
                y = F.relu(self.convs[i - 1](x))
                y = self.bns[i - 1](y)
                if i > 0 and i % 2 == 0:
                    x = y + old_x  # Residual connection
                    old_x = x
                else:
                    x = y

        # Reshape and average across time dimension
        x = x.view(x.size(0), x.size(1), -1)  # Shape: (batch, feats, time)
        x = torch.mean(x, dim=2)  # Global average pooling

        return x.unsqueeze(-2), length
