################################################
# Visualization module
###############################################
# This module implements all functionality for
# visualization of data
###############################################
import matplotlib.pyplot as plt
import numpy as np

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
