from __future__ import print_function, division
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, utils, datasets

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


if __name__ =="__main__":

    ##### simple example for data loading: #####
    # first we define the transform done one every image
    data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # next we load the dataset
    image_dataset = datasets.ImageFolder(root="./data/project-cholangiocarcinoma/MSIvsMSS/Train",transform=data_transform)
    # and prepare everything for training
    dataset_loader = DataLoader(image_dataset,batch_size=4, shuffle=True, num_workers=4)


    plt.imshow(image_dataset.__getitem__(110)[0].numpy().transpose(1, 2, 0))