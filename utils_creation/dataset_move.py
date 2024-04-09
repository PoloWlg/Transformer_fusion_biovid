import cv2
import sys
import os
import csv
# import pandas as pd

# from PIL import Image
import os
import shutil

bl = 'BL'
one= 'PA1'
two= 'PA2'
three= 'PA3'
four= 'PA4'

glob_src = '/projets/AS84330/Datasets/Biovid/PartA/biovid_physio_A/biosignals_filtered'
glob_dest = '/projets/AS84330/Datasets/Biovid/PartA/biovid_physio_A/biosignals_filtered_moved'

#loop to go through all the subdirectories in the source directory

for sub_dir in os.listdir(glob_src):
    sub_dir_path = os.path.join(glob_src, sub_dir)
    print(sub_dir_path)
    #loop to go through all the subdirectories in the subdirectories
    for sub_sub_dir in os.listdir(sub_dir_path):
        print(sub_sub_dir)
        source_path = os.path.join(glob_src,sub_dir,sub_sub_dir) #'/home/livia/work/Biovid/PartB/sub_img_red_classes/081714_m_36/081714_m_36-BL1-081'
        if bl in sub_sub_dir:
            destination_path = os.path.join(glob_dest,'0',sub_sub_dir) #'/home/livia/work/Biovid/PartB/biovid_classes/0/081714_m_36-BL1-081'
        elif one in sub_sub_dir:
            destination_path = os.path.join(glob_dest,'1',sub_sub_dir)
        elif two in sub_sub_dir:
            destination_path = os.path.join(glob_dest,'2',sub_sub_dir)
        elif three in sub_sub_dir:
            destination_path = os.path.join(glob_dest,'3',sub_sub_dir)
        elif four in sub_sub_dir:
            destination_path = os.path.join(glob_dest,'4',sub_sub_dir)
        shutil.copytree(source_path, destination_path)