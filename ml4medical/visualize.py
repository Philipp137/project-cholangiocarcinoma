################################################
# Visualization module
###############################################
# This module implements all functionality for
# visualization of data
###############################################
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import re
import glob
from torchvision.utils import make_grid
from torchvision import transforms

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

def show_heatmap(xlist,ylist,predictions):
    """
    ** show_heatmap **
    :param xlist: xlist[i] is the x_i coordinate of the tile_i
    :param ylist: ylist[i] is the y_i coordinate of the tile_i
    :param predictions: predictions[i] is the prediction of the tile_i
    :return: figure handle
    """
    map = np.NaN * np.zeros([max(xlist) + 1, max(ylist) + 1])
    map[xlist, ylist] = predictions
    plt.figure()
    cmap = plt.get_cmap("seismic")
    cmap
    h = plt.imshow(map.T, cmap=plt.get_cmap("seismic"), vmin=0, vmax=1)
    plt.xticks([])
    plt.yticks([])
    return h




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


def show_transforms(data_module, samples_per_class, samples_per_row, variant='CCC', case='train'):
    ds = data_module.train_dataset if case=='train' else data_module.val_dataset
    train_class_0_idxs = np.random.choice(ds.idx_in_class[0], samples_per_class)
    train_class_1_idxs = np.random.choice(ds.idx_in_class[1], samples_per_class)

    train_c0_samples = make_grid(ds[train_class_0_idxs][0].clone(), samples_per_row)
    train_c1_samples = make_grid(ds[train_class_1_idxs][0].clone(), samples_per_row)
    ds.apply_transforms = False
    ds.apply_default_transforms = True
    train_c0_samples_orig = ds[train_class_0_idxs][0]
    train_c1_samples_orig = ds[train_class_1_idxs][0]
    # if variant == 'CCC':
    #     train_c0_samples_orig = ds.default_transforms(train_c0_samples_orig)
    #     train_c1_samples_orig = ds.default_transforms(train_c1_samples_orig)
    train_c0_samples_orig = make_grid(train_c0_samples_orig, samples_per_row)
    train_c1_samples_orig = make_grid(train_c1_samples_orig, samples_per_row)
    ds.apply_transforms = True
    ds.apply_default_transforms = True
    
    all_samples = make_grid(torch.stack([train_c0_samples, train_c1_samples, train_c0_samples_orig, train_c1_samples_orig]), 2)
    plt.figure()
    plt.imshow(all_samples.permute([1, 2, 0]))
    
    
if __name__ =="__main__":
    project_directory = '/run/media/phil/Elements/data/CCC//14-51098/'
    tile_list = sorted(glob.glob(project_directory+'**/*.png', recursive=True))

    xlist , ylist, tile_num_list = [], [], []
    for tile_path in tile_list:
        file = os.path.basename(tile_path)
        m = re.match("(\d{2})-(\d+)_(\d+)_(\d+)_(\d+).[png]*",file)
        year = int(m.group(1))
        id = int(m.group(2))
        tile_number = int(m.group(3))
        x = int(m.group(4))
        y = int(m.group(5))
        print(f'tile: {file} year: 20{year} id: {id} (x,y) = {x,y}' )
        xlist.append(x)
        ylist.append(y)
        tile_num_list.append(tile_number)

    show_heatmap(xlist,ylist,predictions = tile_num_list)