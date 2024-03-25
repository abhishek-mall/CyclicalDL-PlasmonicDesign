import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import torchvision
import torchvision.utils as vutils
from torch.distributions import normal

import glob
import numpy as np
from PIL import Image

import os
from random import shuffle
import pandas as pd


#number of Channels
nc = 1

#Network Structure Parameters
kernel_size = 4
stride = 2
padding = 1


#Generator

def denormalize(input):
    out = (input+1)*255/2
    return out 

class Generator(nn.Module): 

    def __init__(self): 
        super(Generator, self).__init__() 
        self.model = nn.Sequential( 
            nn.ConvTranspose2d(1024, 512, 4, 1, 0, bias = False), 
            nn.BatchNorm2d(512), 
            nn.ReLU(True), 
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False), 
            nn.BatchNorm2d(256), 
            nn.ReLU(True), 
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False), 
            nn.BatchNorm2d(128), 
            nn.ReLU(True), 
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias = False), 
            nn.Tanh() 
        )

    def forward(self, input): 
        output = self.model(input) 
        output = denormalize(output)
        return output

#Discriminator
        
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(nc, 64, kernel_size, stride, padding)
        self.bn11 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size, stride, padding)
        self.bn12 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size, stride, padding)
        self.bn13 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size, stride, padding)
        self.bn14 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 100, 4, 2, 1)
        self.bn15 = nn.BatchNorm2d(100)
        self.fc1 = nn.Linear(501, 512)
        self.bn21 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
       
        
    def forward(self, x, spctr):
        
        x = self.conv1(x)
        x = self.bn11(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn12(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn13(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn14(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn15(x)
        x = self.relu(x)

        
        x = torch.flatten(x, 1)
        spctr = spctr.view(spctr.shape[0], -1)
        x = torch.cat((x, spctr), 1)

        x = self.fc1(x)
        x = self.bn21(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x