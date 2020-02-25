import torch
import torch.nn as nn
import torch.nn.functional as F

from ops.spectral_norm import spectral_norm as SpectralNorm
from ops.function import adaptive_instance_normalization as adain

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
            layers_makeup.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=2, padding=1, bias=False))
            layers_makeup.append(nn.InstanceNorm2d(curr_dim, affine=True))
            layers_makeup.append(nn.ReLU(inplace=True))
            
        layers_makeup.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers_makeup.append(nn.Flatten())
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
        curr_dim = input_dim

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
        
        # Up-Sampling
        layers = []
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2
        
        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(3, affine=True))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, r_b, r_m):
        makeupFeature = self.makeup(r_m)

        out = self.res_1(r_b)
        out = adain(out, makeupFeature)
        out = self.res_2(out)
        out = adain(out, makeupFeature)
        out = self.res_3(out)
        out = adain(out, makeupFeature)
        out = self.res_4(out)
        out = adain(out, makeupFeature)

        out = self.main(out)

        return out
        

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result



# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)