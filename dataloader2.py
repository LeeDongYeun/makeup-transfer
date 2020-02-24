from torchvision import transforms
from torch.utils.data import DataLoader
from data_loaders.makeup import MAKEUP2
import torch
import numpy as np
import PIL

def get_loader(data_config, config, mode="train"):
    # return the DataLoader
    dataset_name = data_config.name
    transform = transforms.Compose([
    transforms.Resize(config.img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    transform_mask = transforms.Compose([
        transforms.Resize(config.img_size, interpolation=PIL.Image.NEAREST),
        ToTensor])
    print(config.data_path)