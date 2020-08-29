import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''Discriminator'''
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.seq = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        input = input.view(input.size(0), -1)
        result = self.seq(input)
        return result


'''Generator'''
class Generator(nn.Module):
    def __init__(self, latent_size):
        super(Generator, self).__init__()

        self.seq = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, input):
        result = self.seq(input)
        result = result.view(-1, 1, 28, 28)
        return result