import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# from torch.utils.tensorboard import SummaryWriter

class Discriminator(nn.Module):
  def __init__(self, img_dim):
    super().__init__()

    self.disc = nn.Sequential(
        nn.Linear(img_dim, 128), # input of 784 ( the generated image ) and out of 128
        nn.LeakyReLU(0.1),
        nn.Linear(128, 1), # input of 128 and out of 1
        nn.Sigmoid(), # sigmoid to determinate if it's real or not :p
    )

  def forward(self, x):
    return self.disc(x)
  
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256), # input of 64 (the noise) and out of 256
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim), # input of 256 and out of 784
            nn.Tanh(),  # normalize inputs to [-1, 1] so make outputs [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)
    
