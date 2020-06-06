import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(3, 3, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def auto_encoder():
    return Autoencoder()
