import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
import torchvision.utils as vutils
from torch.distributions import normal
import glob
import cv2
import numpy as np
from PIL import Image
import os
from random import shuffle
import pandas as pd
from model_CGAN import Generator, Discriminator
from model_SNN import SNN
import math

#The device to run on
use_cuda = 0
device = torch.device("cuda" if use_cuda else "cpu")
batchsize = 1000

#Threshold for MAE
threshold = 0.13

def getDesiredSpctr():
	array = np.zeros((batchsize, 101))

	#Parameters to generate desired spectrum
	for b in range(0, batchsize):
		f0 = np.random.randint(20, 40)
		f1 = np.random.randint(50, 70)
		f2 = np.random.randint(80, 90)
		sigma0 = np.random.randint(1, 10)
		sigma1 = np.random.randint(1, 10)
		sigma2 = np.random.randint(1, 10)

		for f in range(0, 101):
			val = 0.5* math.exp(-((f-f0)**2)/(2*(sigma0**2))) + 0.5* math.exp(-((f-f1)**2)/(2*(sigma1**2))) + 0.7* math.exp(-((f-f2)**2)/(2*(sigma2**2)))
			array[b][f] = val
	array = torch.tensor(array)
	return array


def MAE(real, generated):
    diff = abs(real - generated)
    diff = np.mean(diff)   
    return diff

#Resizes generator output into (32, 32)
def resizeImages(genImgs):
	imgs = genImgs.detach().numpy()

	imgsResized = np.zeros((batchsize, 1, 32, 32))

	for i in range(0, imgs.shape[0]):
		imgsResized[i][0] = cv2.resize(imgs[i][0], (32, 32))

	return torch.tensor(imgsResized)


#Checkpoint loading function
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = True
    model.eval()
    return model


#Loading all the three models 
generator = load_checkpoint("Weights/gen7980.pth").to(device)
discriminator = load_checkpoint("Weights/dis7980.pth").to(device)
snn = load_checkpoint("WeightsSNN/455snn.pth").to(device)


#Passes spectrum concatenated with noise through generator
def generatorOutput(spctr):
	spctr = spctr.to(device)
	spctr = spctr.float()
	spctr = spctr.view(batchsize, spctr.shape[1], 1, 1)
	z = torch.cat((torch.rand(batchsize, 411, 1, 1).to(device), spctr), 1)
	genImgs = generator(z) 
	return genImgs

#Passing image to SNN to get spectrum
def SNNOutput(imgs):
	imgs = imgs.to(device)
	imgs = imgs.float()
	output = snn.forward(imgs) 
	return output


desiredSpctr = getDesiredSpctr()
genImgs = generatorOutput(desiredSpctr)
snnInput = resizeImages(genImgs)
snnOut = SNNOutput(snnInput)

desiredSpctr = desiredSpctr.cpu().numpy()
snnOut = snnOut.cpu().detach().numpy()

for i in range(0, snnOut.shape[0]):
	diff = MAE(desiredSpctr[i], snnOut[i])
	if(diff < threshold):
		save = genImgs[i].view(genImgs[i].shape[1], genImgs[i].shape[2], genImgs[i].shape[0])
		save = save.cpu().detach().numpy()
		cv2.imwrite("Experiment_Generated_Images/" + str(i+1)+".png", save)
		np.savetxt("Experiment_Spectrum/" + str(i+1) + "desired.txt", desiredSpctr[i])
		np.savetxt("Experiment_Spectrum/" + str(i+1) + "generated.txt", snnOut[i])


