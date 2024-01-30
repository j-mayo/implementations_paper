import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

# Some blocks

class ResBlock2d(nn.Module):
    def __init__(self, channels, kernel_size, stride, use_dropout, use_bias, padding):
        super(ResBlock2d, self).__init__()
        self.conv_block = []
        self.conv_block.append(nn.Conv2d(channels, channels, kernel_size, stride, bias=use_bias, padding=padding))
        if use_dropout > 0:
            self.conv_block.append(nn.Dropout(use_dropout))
        self.conv_block.append(nn.Conv2d(channels, channels, kernel_size, stride, bias=use_bias, padding=padding))
        
        self.conv_block = nn.Sequential(*self.conv_block)
        
        
    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResNorm2d(nn.Module):
    def __init__(self, channels, kernel_size, stride, use_dropout=0, padding=0, use_bias=False, norm_type=nn.InstanceNorm2d, activation_type=nn.ReLU, activation_value=True):
        super(ResNorm2d, self).__init__()
        self.res = ResBlock2d(channels, kernel_size, stride, use_dropout, use_bias, padding)
        self.norm = norm_type(channels)
        if activation_type != nn.ReLU:
            self.activation = activation_type(activation_value, inplace=True)
        else:
            self.activation = activation_type(activation_value)
        
        self.model = nn.Sequential(self.res, self.norm, self.activation)
        
    
    def forward(self, x):
        #print(x.shape)
        return self.model(x)
        

class ConvNorm2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, use_dropout=0, use_bias=False, norm_type=nn.InstanceNorm2d, activation_type=nn.ReLU, activation_value=True):
        super(ConvNorm2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=use_bias, padding=padding)
        self.norm = norm_type(out_channels)
        if activation_type != nn.ReLU:
            self.activation = activation_type(activation_value, inplace=True)
        else:
            self.activation = activation_type(activation_value)
        
        self.model = nn.Sequential(self.conv2d, self.norm, self.activation)
        
    
    def forward(self, x):
        #print(x.shape)
        return self.model(x)
    

class UpConvNorm2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, output_padding=0, use_dropout=0, use_bias=False, norm_type=nn.InstanceNorm2d, activation_type=nn.ReLU, activation_value=True):
        super(UpConvNorm2d, self).__init__()
        self.upconv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, bias=use_bias, padding=padding, output_padding=output_padding)
        self.norm = norm_type(out_channels)
        if activation_type != nn.ReLU:
            self.activation = activation_type(activation_value, inplace=True)
        else:
            self.activation = activation_type(activation_value)
        
        self.model = nn.Sequential(self.upconv2d, self.norm, self.activation)

    
    def forward(self, x):
        #print(x.shape)
        return self.model(x)



# Generator

class Generator(nn.Module):
    def __init__(self, ngf, nc, img_size=(3, 128, 128), norm_type=nn.InstanceNorm2d, use_dropout=0):
        super(Generator, self).__init__()
        self.img_size = img_size # 1 x 28 x 28 for MNIST
        self.ngf = ngf
        self.nc = nc
        self.use_bias = (norm_type == nn.InstanceNorm2d)
        self.norm_type = norm_type
        
        if img_size[-1] == 128:
            self.get_resnet(in_channels=self.img_size[0], down=2, mid=6, use_dropout=use_dropout)
        elif img_size[-1] == 256:
            self.get_resnet(in_channels=self.img_size[0], down=2, mid=9, use_dropout=use_dropout)
        else:
            raise AttributeError
        

    def get_resnet(self, in_channels, down, mid, use_dropout):
        self.model = []
        out_channels = self.ngf
        self.model.append(nn.ReflectionPad2d(3)) # paper
        self.model.append(ConvNorm2d(in_channels, out_channels, 7, 1, padding=0, use_dropout=use_dropout, use_bias=self.use_bias, norm_type=self.norm_type))
        # downsampling
        for i in range(1, down+1):
            self.model.append(ConvNorm2d(out_channels, out_channels*2, 3, 2, padding=1, use_dropout=use_dropout, use_bias=self.use_bias, norm_type=self.norm_type))
            out_channels *= 2
        
        # res
        for i in range(1, mid+1):
            self.model.append(ResNorm2d(out_channels, 3, 1, use_dropout=use_dropout, use_bias=self.use_bias, norm_type=self.norm_type, padding=1))
        # TODO - add reflection padding at ResNorm2d
        # upsampling
        for i in range(1, down+1):
            self.model.append(UpConvNorm2d(out_channels, out_channels//2, 3, 2, padding=1, output_padding=1, use_dropout=use_dropout, use_bias=self.use_bias, norm_type=self.norm_type))
            out_channels //= 2
        
        self.model.append(nn.ReflectionPad2d(3)) # paper
        self.model.append(nn.Conv2d(out_channels, self.nc, 7, 1, bias=self.use_bias, padding=0))
        self.model.append(nn.Tanh())
        
        self.model = nn.Sequential(*self.model)
    
    def forward(self, x):
        output = self.model(x)
        return output

class Discriminator(nn.Module):
    def __init__(self, ngf, nc, img_size=(3, 128, 128), norm_type=nn.InstanceNorm2d, use_dropout=0):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.ngf = ngf
        self.nc = nc
        self.use_bias = (norm_type == nn.InstanceNorm2d)
        self.norm_type = norm_type
        
        self.get_model(img_size[0], 4, use_dropout)
        
    
    def get_model(self, in_channels, n, use_dropout):
        self.model = []
        # self.model.append(nn.Conv2d(in_channels, self.ngf, 4, 2, padding=1))
        # self.model.append(nn.LeakyReLU(0.2, True))
        #in_channels = self.ngf
        out_channels = self.ngf
        for _ in range(n-1):
            self.model.append(ConvNorm2d(in_channels, out_channels, 4, 2, use_bias=self.use_bias, padding=1, norm_type=self.norm_type, activation_type=nn.LeakyReLU, activation_value=0.2))
            in_channels = out_channels
            out_channels *= 2
        
        self.model.append(ConvNorm2d(in_channels, in_channels, 4, 1, use_bias=self.use_bias, padding=1, norm_type=self.norm_type, activation_type=nn.LeakyReLU, activation_value=0.2))
        
        self.model.append(nn.Conv2d(in_channels, 1, 4, 1, padding=1))
        self.model = nn.Sequential(*self.model)
    
    def forward(self, img):
        output = self.model(img)
        return output