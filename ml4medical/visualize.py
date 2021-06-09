################################################
# Visualization module
###############################################
# This module implements all functionality for
# visualization of data
###############################################
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

# Taken from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def imshow4normalized(inp, title=None, mean = np.array([0.485, 0.456, 0.406]), std = np.array([0.229, 0.224, 0.225])):
    """This function reverts normalization and plots input tensor"""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean # at this point we reverse the normalization
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def show_datasamples(data_module, samples_per_class, samples_per_row):
    train_class_0_idxs = np.random.choice(data_module.train_dataset.idx_in_class[0], samples_per_class)
    train_class_1_idxs = np.random.choice(data_module.train_dataset.idx_in_class[1], samples_per_class)
    val_class_0_idxs = np.random.choice(data_module.val_dataset.idx_in_class[0], samples_per_class)
    val_class_1_idxs = np.random.choice(data_module.val_dataset.idx_in_class[1], samples_per_class)
    
    train_c0_samples = make_grid(data_module.train_dataset[train_class_0_idxs][0], samples_per_row)
    train_c1_samples = make_grid(data_module.train_dataset[train_class_1_idxs][0], samples_per_row)
    val_c0_samples = make_grid(data_module.val_dataset[val_class_0_idxs][0], samples_per_row)
    val_c1_samples = make_grid(data_module.val_dataset[val_class_1_idxs][0], samples_per_row)
    
    all_samples = make_grid(torch.stack([train_c0_samples, train_c1_samples, val_c0_samples, val_c1_samples]), 2)
    plt.figure(8)
    plt.imshow(all_samples.permute([1, 2, 0]))