#      Aachen Project 

This repository contains source files of machine learning tools for
medical tumor classification.

----------------------------------------------------------------------

## Qupath Workflow

1. Download Qupath from https://github.com/qupath/qupath/releases/tag/v0.2.3 and unzip
2. Run Qupath by calling the binary file
3. Create Folder: mkdir project_aachen
4. in qupath file>create project and choose project_aachen
5. in qupath file>add image choose the tif (filename for example: 16-08510.tif)
6. now qupath is exporting some data in project_aachen
    there you find a folder named "data"
7. copy the annotations from the source folder into the data folder to see the annotations in qupath

## Python

Make sure your python version matches the requirements listed in ''requirements.txt''!
To make your life easy, I included ''conda-env.yml'' to match all python requirements
with the help of conda. Simply create a new conda environment:

    conda env create -f conda-env.yml 

and activate it:

    conda activate conda-env

After all requirements match you may be able to execute our source code.
All source files of the packages are included in the directory
    
    ./ml4medical/

You can execute the individual routines via in the folder via
    
    import ml4medical 
    ml4medical.myroutine()
    
 or 
 
    form ml4medical import myroutine
