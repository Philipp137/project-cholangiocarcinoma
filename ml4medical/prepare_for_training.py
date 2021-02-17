'''
This script has the following purpose:
1. take the images from the source_path and generate tiles from it using PyHIST (https://pyhist.readthedocs.io/en)
    or from the annotations in qupath
2. sort the tiles according to "yes" and "no" in mappe1.xlsx in two folders
        prefix: data/
        train/pos/ and train/neg/
        test/pos/ and test/neg

'''

import glob
import os
import pandas as pd
import numpy as np

## Define the folders and files
IO_paths = {"source": '/run/media/phil/Elements/iCCA/',          # raw images, i.e. *tiff files
           "annotations": '/run/media/phil/Elements/qupath/',   # folder with annotations of pathologist
           "labels": '/run/media/phil/Elements/CCC-labels.csv', # csv data with id of the tiff image and label (pos, neg)
           "work" : '/run/media/phil/Elements/work/',           # work folder for intermediate results and files
           "ML-data":   '/run/media/phil/Elements/data/',}       # data for neuronal network to train and test

##
fraction_train = 0.8

## choose a method to export tiles
    # method = 'pyhist' uses pyhist for image segementation and exporting tiles without human help (pathologist)
    # method = 'qupath' exports tiles from pathologist defined areas
method = ""

if method == "pyhist":
    ## loop over all scans in the source folder
    i = 0
    for filename in glob.glob(IO_paths["source"]+'**/*.tif', recursive=True):
        i = i + 1
        # mask the images and export tiles
        command = 'python PyHIST/pyhist.py --save-patches --save-tilecrossed-image --borders 0000 --corners 1010 --percentage-bc 1 ' + \
                  '--content-threshold 0.4 --patch-size 512 --output-downsample 4 --k-const 1000 --minimum_segmentsize 1000 --method="adaptive" ' + \
                  '--output %s %s'%(IO_paths['work'], filename)
        success = os.system(command)
        if success !=0: print("Segmentation of %s did not work! Stopping script" % filename); break

elif method == "qupath":
    pass
    # 1. create folder for each scan (*.tif)

    # 2. put in the basic blueprint of a qupath project
    # (with specific folder structure project.qpproj file etc.)

    # 3. copy annotations for each image in the folder

    # 4. run groovy script to export tiles from the annotations


df = pd.read_csv(IO_paths["labels"])
test = {"pos": IO_paths["ML-data"] + "/test/pos/", "neg": IO_paths["ML-data"] + "/test/neg/"}
train = {"pos": IO_paths["ML-data"] + "/train/pos/", "neg": IO_paths["ML-data"] + "/train/neg/"}

for f in list(test.values())+list(train.values()):
    if not os.path.exists(f):
        os.makedirs(f)
    else:
        assert(len(os.listdir(f))==0) , "Directory %s is not empty! Please remove files!" % f

#os.mkdir()

for folder in glob.glob(IO_paths['work']+'*'):
    folder_name = folder.split('/')[-1]  #assuming that folder_name is the same as the id of the patient
    label = df.label[df.scan==folder_name].item()
    if label == "missing":    continue
    print("\n\nscan: %s    label: %s" %(folder_name, label))
    print("Copy tiles to: ")
    for tile in glob.glob(folder+"/"+folder_name+"_tiles/*.png"):
        if np.random.random() < fraction_train:
            print("Tile: %s -> %s" %(tile.split("/")[-1], train[label]))
            command = "cp "+ tile + ' ' + train[label]
        else:
            print("Tile: %s -> %s"%(tile.split("/")[-1], test[label]))
            command = "cp "+ tile + ' ' + test[label]

        os.system(command)
