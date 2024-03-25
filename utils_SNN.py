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
import cv2

#UTILITY FUNCTIONS

#To load saved checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = True
    model.eval()
    return model

def saveResults(real, generated, real_imgs, epoch, dir, imgDir):

    for b in range(0, real.shape[0]):
        out = real[b].detach().numpy()
        file_name = dir+str(epoch)+"_"+str(b)+"_Real"+".csv"
        np.savetxt(file_name, out, delimiter=",")
        
    for b in range(0, generated.shape[0]):
        out = generated[b].detach().numpy()
        file_name = dir+str(epoch)+"_"+str(b)+"_Generated"+".csv"
        np.savetxt(file_name, out, delimiter=",")

    for im in range(0, real_imgs.shape[0]):
        name_gen = imgDir + str(epoch) + "_" + str(im) + ".png"
        gen_imgs_save = real_imgs[im].view(real_imgs[im].shape[1], real_imgs[im].shape[2], real_imgs[im].shape[0])
        gen_imgs_save = gen_imgs_save.cpu().detach().numpy()
        cv2.imwrite(name_gen, gen_imgs_save)

