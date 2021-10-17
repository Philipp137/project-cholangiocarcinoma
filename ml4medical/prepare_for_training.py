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
import os,re
import pandas as pd
import numpy as np
import pathlib
from qupath_utils import create_qupath_project_file, create_qupath_classes_file
from normalize import normalizeStaining
from PIL import Image
from shutil import copyfile
from joblib import Parallel, delayed



def reformat_name(scan_name):
    try:
        id , year, version = scan_name.split('_')
        return year + '-' + id + '-' + version
    except:
        id, year= scan_name.split('_')
        return year + '-' + id


def is_file_ok(file_path, white_pixel_fraction = 0.6):
            img = np.array(Image.open(file_path))

            # check sqaurness:
            h,w,c = img.shape
            if not h  ==  w:
                print("shapes differ:",file_path.split("/")[-1])
                return False
            # check if to white
            number_white_pixels = np.sum(np.mean(img,axis=2)>220)
            number_pixels = h*w
            if number_white_pixels/number_pixels > white_pixel_fraction:
                print("too white:", file_path.split("/")[-1])
                return False
            # check if normalization works
            try:
                normalizeStaining(img=img,
                          Io=240,
                          alpha=1,
                          beta=0.15)
            except:
                print("normalization error:", file_path.split("/")[-1])
                return False


            return True


def check_file(subdir,file):
    file_path = os.path.join(subdir, file)
    if is_file_ok(file_path):
        # now calculate smallest x,y coordinate
        #  - filename is like: '10049_13_1.4 - 2021-04-14 17.51.37 [x=12361,y=33713,w=1124,h=1124].png'
        #  - re.findall(r'(\w+)=(\d+)', file) is a regexp to find the values and keys seperated by =
        #  - dict converts the return to a dictionary
        #  - and map(int, list) converts all elements in list to integer
        xpos, ypos, w, h = map(int, dict(re.findall(r'(\w+)=(\d+)', file)).values())


        return file, int(xpos),int(ypos),int(w),int(h)



def check_parallel(files,subdir, n_jobs=4):
    is_ok = lambda file: check_file(subdir, file)
    checked_list = Parallel(n_jobs=n_jobs)(map(delayed(is_ok), files))


    clean_list = list(filter(None.__ne__, checked_list)) # this filters all the none elements
    clean_list = np.asarray(clean_list)
    clean_files = list(np.asarray(list(filter(None.__ne__, clean_list)))[:,0])
    xmin = min(np.asarray(list(np.asarray(list(filter(None.__ne__, clean_list)))[:, 1]),dtype=np.int))
    ymin = min(np.asarray(list(np.asarray(list(filter(None.__ne__, clean_list)))[:, 2]),dtype=np.int))
    w = np.asarray(list(filter(None.__ne__, clean_list)))[:, 3]
    assert( np.all(w==w[0]) ), "all tiles must have same size/width!!!"
    return clean_files, xmin, ymin, int(w[0])




## Define the folders and files
cohort = "mainz"
if cohort=="aachen":
    IO_paths = {"source": '/run/media/phil/Elements/iCCA/',          # raw images, i.e. *tiff files
           "annotations": '/run/media/phil/Elements/qupath/',   # folder with annotations of pathologist
           "labels": '/run/media/phil/Elements/CCC-labels.csv', # csv data with id of the tiff image and label (pos, neg)
           "work" : '/run/media/phil/Elements/work_qupath/',           # work folder for intermediate results and files
           "ML-data":   '/run/media/phil/Elements/data/',}       # data for neuronal network to train and test
elif cohort == "mainz":
    IO_paths = {"tiles_raw": '/run/media/phil/Elements/mainz/tiles_raw/', # tiles after running qupath script on project.qproj
                "tiles_clean": '/run/media/phil/Elements/mainz/tiles_clean/',  # tile directory after cleaning 
                "tiles_dirty": '/run/media/phil/Elements/mainz/tiles_dirty/'}  # tile directory after cleaning 

if cohort == "aachen":
    for filename in sorted(glob.glob(IO_paths["source"]+'**/*.tif', recursive=True)):
        i = i + 1
        # %-----------------------------------------------------------
        # 1. create folder for each scan (*.tif) with pathology number
        file = os.path.basename(filename)                       # filename of tiff without path
        ID, extension = os.path.splitext(file)                  # filename split into name and extension .tiff
        project_directory = IO_paths["work"] + ID
        if not os.path.exists(project_directory):
            os.makedirs(project_directory)
            print(f"Creating project dir in: {project_directory:s}")
        # %-----------------------------------------------------------
        # 2. put in the basic blueprint of a qupath project
        # (with specific folder structure project.qpproj file etc.)
        # 2a folder structure:
        pathlib.Path(project_directory+'/data').mkdir(parents=True, exist_ok=True)
        pathlib.Path(project_directory + '/data/2/').mkdir(parents=True, exist_ok=True)
        pathlib.Path(project_directory + '/data/1/').mkdir(parents=True, exist_ok=True)
        pathlib.Path(project_directory + '/classifiers/').mkdir(parents=True, exist_ok=True)
        # 2b create the project file "project.qproj" in project_directory and include the path to the tiff file
        txt, success = create_qupath_project_file(project_directory, filename, file)
        # 2c create json file of classes (stroma , tumor, etc.)
        create_qupath_classes_file(project_directory)
        assert (success > 0) , " no project file written"
        # %-----------------------------------------------------------
        # 3. copy annotations for each image in the folder
        annotations= IO_paths["annotations"] + "/" + str(i)+"/*"
        command = "cp %s %s" %(annotations, project_directory+'/data/1/')
        success = os.system(command)

    # 4. run groovy script to export tiles from the annotations
elif cohort == "mainz":
        print("mainz")
        nscans = 0
        for subdir, dirs, files in os.walk(IO_paths["tiles_raw"]):

            if len(files)>0: # check if dir is empty
                nscans = nscans + 1
                #if nscans <= 51:
                #    continue
                scan_name = subdir.split("/")[-1]
                scan_name = reformat_name(scan_name)
                print("#####################################")
                print("%d scan: "%(nscans), scan_name)
                print("#####################################")

                newdir = os.path.join(IO_paths["tiles_clean"],scan_name)
                pathlib.Path(newdir).mkdir(parents=True, exist_ok=True)

                clean_files, xmin, ymin, width = check_parallel(files,subdir)

                print("number raw tiles:", len(files))
                print("number clean tiles:", len(clean_files))
                for i, file in enumerate(clean_files):
                    xpos, ypos, width, h = map(int, dict(re.findall(r'(\w+)=(\d+)', file)).values())
                    x_rel = int((xpos-xmin)/width)
                    y_rel = int((ypos-ymin)/width)
                    new_file = scan_name +'_'+str(i)+'_'+str(x_rel)+'_'+str(y_rel)+'.png'
                    new_path = os.path.join(newdir,new_file)
                    file_path = os.path.join(subdir, file)
                    copyfile(file_path , new_path)

                dirty_files = list(set(files)-set(clean_files))
                newdir = os.path.join(IO_paths["tiles_dirty"],scan_name)
                pathlib.Path(newdir).mkdir(parents=True, exist_ok=True)
                for i,file in enumerate(dirty_files):
                    file_path = os.path.join(subdir, file)
                    new_path = os.path.join(newdir, file)
                    copyfile(file_path , new_path)


