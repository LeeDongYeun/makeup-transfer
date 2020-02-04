import torch
import torch.nn as nn
import torch.nn.functional as F

from ops.spectral_norm import spectral_norm as SpectralNorm

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input

class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)


class Encoder(nn.Module):
    """Encoder."""
    def __init__(self, conv_dim=64):
        super(Encoder, self).__init__()
        input_dim = 3

        layers = []
        layers.append(nn.Conv2d(input_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=3, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2
        
        self.main = nn.Sequential(*layers)

        # Base representation encoder
        layers_base = []
        for i in range(4):
            layers_base.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        
        self.base = nn.Sequential(*layers_base)

        # Makeup representation encoder
        layers_makeup = []
        for i in range(2):
            layers_makeup.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=3, stride=2, padding=1, bias=False))
            layers_makeup.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers_makeup.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2
        layers_makeup.append(nn.MaxPool2d(kernel_size=1, stride=1))
        layers_makeup.append(nn.Linear(curr_dim, 32))

        self.makeup = nn.Sequential(*layers_makeup)

    def forward(self, x):
        out = self.main(x)
        out_base = self.base(out)
        out_makeup = self.makeup(out)

        return out_base, out_makeup
    

class Generator(nn.Module):
    """Generator."""
    def __init__(self, conv_dim=64):
        super(Generator, self).__init__()
        input_dim = 256

        # Makeup representation fully connected layer
        layers_makeup = []
        layers_makeup.append(nn.Linear(32, 256))
        layers_makeup.append(nn.ReLU(inplace=True))
        layers_makeup.append(nn.Linear(256, 256))
        layers_makeup.append(nn.ReLU(inplace=True))
        layers_makeup.append(nn.Linear(256, 512))

        self.makeup = nn.Sequential(*layers_makeup)

        # Main Generator
        self.res_1 = ResidualBlock(dim_in=input_dim, dim_out=input_dim)
        self.res_2 = ResidualBlock(dim_in=input_dim, dim_out=input_dim)
        self.res_3 = ResidualBlock(dim_in=input_dim, dim_out=input_dim)
        self.res_4 = ResidualBlock(dim_in=input_dim, dim_out=input_dim)

        layers = []
        







