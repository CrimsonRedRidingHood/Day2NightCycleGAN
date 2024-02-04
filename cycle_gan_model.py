import os
import math
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torchvision.io import read_image
from PIL import Image
import numpy as np

MODEL_NAME = 'gen.nn'
IMAGE_SIZE = 128
GENERATOR_ENCODER_BLOCKS = 2 # all trained models used 2 encoder blocks, so don't change unless you trained the model yourself
GENERATOR_INITIAL_CONV_OUT = 64 # don't change this
BATCH_NORM_USAGE_THRESHOLD = 32

batch_size = 1
generator_residual_blocks = 6 if IMAGE_SIZE < 256 else 9
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
use_batchnorm = batch_size >= BATCH_NORM_USAGE_THRESHOLD
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
transform=tt.Compose([
        tt.Resize(IMAGE_SIZE),
        tt.CenterCrop(IMAGE_SIZE),
        tt.ToTensor(),
        tt.Normalize(*stats)])

class Generator(nn.Module):
    def add_norm_relu(self, layers, features, batch_size, use_batchnorm):
        if batch_size == 1:
            layers += [nn.InstanceNorm2d(features)]
        elif use_batchnorm:
            layers += [nn.BatchNorm2d(features)]
        layers += [nn.ReLU(True)]
    
    def __init__(self, image_size, batch_size, use_batchnorm, encoder_blocks, residual_blocks):
        super().__init__()
        in_layer = [nn.ReflectionPad2d(3),
                 nn.Conv2d(3, GENERATOR_INITIAL_CONV_OUT, kernel_size=7, padding=0, bias=not use_batchnorm)]
        self.add_norm_relu(in_layer, GENERATOR_INITIAL_CONV_OUT, batch_size, use_batchnorm)
        
        self.in_layer = nn.Sequential(*in_layer)

        encoder = []
        
        features = GENERATOR_INITIAL_CONV_OUT
        for i in range(encoder_blocks):
            encoder += [nn.Conv2d(features, features * 2, kernel_size=3, stride=2, padding=1, bias=not use_batchnorm)]
            self.add_norm_relu(encoder, features * 2, batch_size, use_batchnorm)
            features *= 2
            
        self.encoder = nn.Sequential(*encoder)

        self.residual_blocks = nn.ModuleList()
            
        for i in range(residual_blocks):
            residual_block = [nn.ReflectionPad2d(1), nn.Conv2d(features, features, kernel_size=3, padding=0, bias=not use_batchnorm)]
            self.add_norm_relu(residual_block, features, batch_size, use_batchnorm)
            
            residual_block += [nn.ReflectionPad2d(1), nn.Conv2d(features, features, kernel_size=3, padding=0, bias=not use_batchnorm)]
            if batch_size == 1:
                residual_block += [nn.InstanceNorm2d(features)]
            elif use_batchnorm:
                residual_block += [nn.BatchNorm2d(features)]
            
            self.residual_blocks.append(nn.Sequential(*residual_block))

        decoder = []
            
        for i in range(encoder_blocks):
            decoder += [nn.ConvTranspose2d(features, features // 2,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=not use_batchnorm)]
            self.add_norm_relu(decoder, features // 2, batch_size, use_batchnorm)
            features //= 2
            
        self.decoder = nn.Sequential(*decoder)
        
        assert(features == GENERATOR_INITIAL_CONV_OUT)
        
        self.out_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(features, 3, kernel_size=7, padding=0), nn.Tanh())
        
    def forward(self, x):
        x = self.in_layer(x)
        x = self.encoder(x)

        for res in self.residual_blocks:
            x = x + res(x)

        x = self.decoder(x)
        x = self.out_layer(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
    
def prepare_image(filename):
    with Image.open(filename) as img:
        width = img.size[0]
        height = img.size[1]
        img = transform(img)
        return img, height, width

def denorm(img):
    return img * stats[1][0] + stats[0][0]
    
def save_result(img, height, width, filename):
    resize_back = tt.Resize((height, width), interpolation=tt.InterpolationMode.BICUBIC)
    img = resize_back(denorm(img))
    save_image(img, filename)
    
def process_image(model, input_filename, output_filename):
    img, height, width = prepare_image(input_filename)
    img = img[None, :]
    img = model(img)
    save_result(img[0], height, width, output_filename)

def load_model():
    model = Generator(IMAGE_SIZE, batch_size, use_batchnorm, GENERATOR_ENCODER_BLOCKS, generator_residual_blocks)
    model.load_state_dict(torch.load(MODEL_NAME))
    model.eval()
    return model