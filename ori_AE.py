import torch
from torch import nn
import numpy as np


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            # [1433] => [300]
            nn.Linear(1433, 1200),
            nn.ReLU(),
            nn.Linear(1200,950),
            nn.ReLU(),
            nn.Linear(950, 300),
            nn.ReLU(),


        )
        self.decoder = nn.Sequential(
            # [300] => [1433]
            nn.Linear(300, 950),
            nn.ReLU(),
            nn.Linear(950, 1200),
            nn.ReLU(),
            nn.Linear(1200, 1433),
            nn.ReLU()

        )

    def forward(self, x):
        # encode
        latent = self.encoder(x)
        # decode
        out = self.decoder(latent)

        return out,latent
