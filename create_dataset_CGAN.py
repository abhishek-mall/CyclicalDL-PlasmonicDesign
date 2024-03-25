import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import torchvision
import torchvision.utils as vutils
from torch.distributions import normal

import numpy as np
from PIL import Image

import os
from random import shuffle
import pandas as pd
import cv2

def data_normalize(tensor):

	center = (torch.max(tensor) + torch.min(tensor))/2.0
	span = (torch.max(tensor) - torch.min(tensor))/2.0
	normalized_tensor = (tensor - center)/span
	 
	return normalized_tensor



def read_data(dir_):
    trainDir = dir_+"/"+ "Train"
    valDir =  dir_+"/"+"Validation"

    imageSize = (64, 64)
    #normalize = transforms.Normalize((0.7, ), (0.7, )) 
    transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(),])

    dataTrain = []
    dataVal = []
    for dir_ in os.listdir(trainDir):
        if(not(dir_.endswith(".pth") or dir_.endswith(".png"))):
            for shapes in sorted(os.listdir(trainDir+"/"+dir_)):
                if(not(shapes.endswith(".png"))):
                    for files in sorted(os.listdir(trainDir+"/"+dir_+"/"+shapes)):
                        if(files.endswith(".png")):
                            img = Image.open(trainDir+"/"+dir_+"/"+shapes+"/"+files)
                            img = transform(img)
                            #img = data_normalize(img)
                            csvfile = trainDir+"/"+dir_+"/"+shapes+"/"+files.split(".")[0]+".csv"
                            spctr = torch.zeros([101, 1, 1])
        
                            files = pd.read_csv(csvfile)
                            arr = files.values
                            array = np.zeros([101, 1])
                            key = files.keys().values
                            array[0][0] = float(key)
                            array[1:101] = arr
                            spctr = torch.tensor(array)
        
                            dataPoint = (img, spctr) 
                            dataTrain.append(dataPoint)

    np.random.shuffle(dataTrain)

    for dir_ in os.listdir(valDir):
        if(not(dir_.endswith(".pth") or dir_.endswith(".png"))):
            for shapes in sorted(os.listdir(valDir+"/"+dir_)):
                if(not(shapes.endswith(".png"))):
                    for files in sorted(os.listdir(valDir+"/"+dir_+"/"+shapes)):
                        if(files.endswith(".png")):
                            img = Image.open(valDir+"/"+dir_+"/"+shapes+"/"+files)
                            img = transform(img)
                            #img = data_normalize(img)
                            
                            csvfile = valDir+"/"+dir_+"/"+shapes+"/"+files.split(".")[0]+".csv"
                            spctr = torch.zeros([101, 1, 1])
        
                            files = pd.read_csv(csvfile)
                            arr = files.values
                            array = np.zeros([101, 1])
                            key = files.keys().values
                            array[0][0] = float(key)
                            array[1:101] = arr
                            spctr = torch.tensor(array)
        
                            dataPoint = (img, spctr) 
                            dataVal.append(dataPoint)

    np.random.shuffle(dataVal)

    return dataTrain, dataVal