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
import cv2
import os
from random import shuffle
from create_dataset_SNN_new  import read_data
import argparse
from model_SNN import SNN
from utils_SNN import load_checkpoint, saveResults
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Passing and parsing arguments
arg_lists = []
parser = argparse.ArgumentParser('Simulation Neural Network')
parser.add_argument('--pathToData',type = str, default = "F:\AE+GAN\Halfwaveplate\CGAN\Revision 1\Data", help = 'Enter path to dataset .')
parser.add_argument('--batchSize',type = int, default = 64 , help = 'Batch size .')
parser.add_argument('--lr',type = float, default = 3e-4, help = 'Learning rate .')
parser.add_argument('--num_epochs',type = int, default = 500 ,help = 'Total number of epochs.')
args, _ = parser.parse_known_args()

use_cuda = 1
device = torch.device("cuda" if use_cuda else "cpu")


#Reading Datasets 
trainData, valData = read_data(args.pathToData)
print("Size of training dataset", len(trainData))
print("Size of validation dataset", len(valData))


#Initialising networks and optimizers
snn = SNN().to(device)
optimizer = torch.optim.Adam(list(snn.parameters()), args.lr, betas=(0.5, 0.999))
criterion = nn.MSELoss()
cosineSim = nn.CosineSimilarity(dim=1)

#Defining a function to training
def train(Traindataset, epoch):
    avgLoss = 0.0
    avgAcc = 0.0

    for i, (imgs, spctr) in enumerate(Traindataset):
        imgs = imgs.to(device)
        imgs = imgs.float()
        spctr = spctr.to(device)
        snn.zero_grad()
        spctr = spctr.squeeze(2)
        output = snn.forward(imgs)
        spctr = spctr.double()
        output = output.double()
        loss = criterion(output.float(), spctr.float())
        accuracy = cosineSim(output, spctr)
        loss.backward()
        optimizer.step()
        #saveResults(spctr.cpu(), output.cpu(), imgs.cpu(), epoch, "spctrTrainSNN/", "fakeTrainSNN/")
        avgLoss += loss.item()
        avgAcc += torch.mean(accuracy).item()

        #avgAcc += cos.item()
        
        
    avgLoss = avgLoss/(i+1)
    avgAcc = avgAcc/(i+1)
    
    return avgLoss, avgAcc

#Defining a function to validation
def validate(Valdataset, epoch):
    avgLoss = 0.0
    avgAcc = 0.0
    for i, (imgs, spctr) in enumerate(Valdataset):
        imgs = imgs.to(device)
        imgs = imgs.float()
        spctr = spctr.to(device)
        snn.zero_grad()
        output = snn.forward(imgs)
        spctr = spctr.squeeze(2)
        loss = criterion(output.float(), spctr.float())
        accuracy = cosineSim(output, spctr)
        saveResults(spctr.cpu(), output.cpu(), imgs.cpu(), epoch , "spctrValSNN/", "fakeValSNN/")
        avgLoss += loss.item()
        avgAcc += torch.mean(accuracy).item()
        #avgAcc += cos.item()
        
    avgLoss = avgLoss/(i+1)
    avgAcc = avgAcc/(i+1)
    
    return avgLoss, avgAcc


if __name__ == '__main__':


    trainLoss = []
    valLoss = []
    trainAcc = []
    valAcc = []  
    trainLossFile = open("trainLoss.txt", "w")
    valLossFile = open("valLos.txt", "w")
    trainAccFile = open("trainAcc.txt", "w")
    valAccFile = open("valAcc.txt", "w")
    for epoch in range(0, args.num_epochs):
        Traindataset = torch.utils.data.DataLoader(trainData, args.batchSize, shuffle = True)
        Valdataset = torch.utils.data.DataLoader(valData, args.batchSize, shuffle = True)
        print("-------------------------------------------------------------------")
        avgLosstrain, avgAcctrain = train(Traindataset, epoch)
        avgLossval, avgAccval = validate(Valdataset, epoch)
        
        #Making an array for accuracy and Loss
        trainLoss.append(avgLosstrain)
        valLoss.append(avgLossval)
        trainAcc.append(avgAcctrain)
        valAcc.append(avgAccval)
        
        trainLossFile.write(str(avgLosstrain))
        trainLossFile.write("\n")
        valLossFile.write(str(avgLossval))
        valLossFile.write("\n")
        
        trainAccFile.write(str(avgAcctrain))
        trainAccFile.write("\n")
        valAccFile.write(str(avgAccval))
        valAccFile.write("\n")

        plt.figure(" loss")
        plt.plot(trainLoss)     
        plt.plot(valLoss)
        
        
        plt.figure("Acc")
        plt.plot(trainAcc)     
        plt.plot(valAcc)
        
        #Let's print something
        
        print("Training loss for epoch number %s is %f" % (epoch+1, avgLosstrain))
        print("validation loss for epoch number %s is %f" % (epoch+1, avgLossval))
        print("Training accuracy for epoch number %s is %f" % (epoch+1, avgAcctrain))
        print("Validation accuracy for epoch number %s is %f" % (epoch+1, avgAccval))
        print("-------------------------------------------------------------------")
        if(epoch and epoch %1 ==0):
            checkpoint = {'model': SNN(),'state_dict': snn.state_dict(), 'optimizer' : optimizer.state_dict()}
            torch.save(checkpoint, 'WeightsSNN/'+str(epoch)+ 'snn.pth')




