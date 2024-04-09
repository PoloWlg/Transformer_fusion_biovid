# import cv2
import sys
import os
import csv
# import pandas as pd

# from PIL import Image
import os
import shutil
import random
from sklearn.model_selection import train_test_split, KFold
from numpy import array



def split_txt_train_val_test():

    # Load your dataset from a text file
    with open('annotations_filtered_peak_2_partA.txt', 'r') as file:
        data = file.readlines()

    # Assuming each line has a label at the end (e.g., 'data... class_id')
    labels = [line.split()[-1] for line in data]

    # Split the data into train, validation, and test sets with equal class distribution
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=69, stratify=labels)
    val_data, test_data, val_labels, test_labels = train_test_split(test_data, test_labels, test_size=0.5, random_state=69, stratify=test_labels)

    # Define file names for the splits
    train_file = 'train.txt'
    val_file = 'val.txt'
    test_file = 'test.txt'

    # Write the data to separate files
    with open(train_file, 'w') as file:
        file.writelines(train_data)

    with open(val_file, 'w') as file:
        file.writelines(val_data)

    with open(test_file, 'w') as file:
        file.writelines(test_data)

    print(f'Dataset has been split and saved into {train_file}, {val_file}, and {test_file}.')

# split_txt_train_val_test()

def split_5folds():

    # Load your dataset from a text file
    with open('annotations_filtered_peak_two_partA.txt', 'r') as file:
        data = array(file.readlines())

    # Assuming each line has a label at the end (e.g., 'data... class_id')
    labels = [line.split()[-1] for line in data]

    # Split the data into train, validation, and test sets with equal class distribution
    # train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for i, (train, test) in enumerate(kf.split(data)):
        train_data = data[train]
        test_data = data[test]
        
        # print(f"  Train: {data[train]}")
        # print(f"  Test:  {data[test]}")
    
        # Define file names for the splits
        train_file = f'train_fold{i+1}.txt'
        test_file = f'test_fold{i+1}.txt'

        # Write the data to separate files
        with open(train_file, 'w') as file:
            file.writelines(train_data)

        with open(test_file, 'w') as file:
            file.writelines(test_data)

    print(f'Dataset has been split and saved into {train_file}, and {test_file}.')

split_5folds()

def create_annotxt():
    peak_2 = True
    
    bl = 'BL'
    one= 'PA1'
    two= 'PA2'
    three= 'PA3'
    four= 'PA4'

    root_dir = '/projets2/AS84330/Datasets/Biovid/PartA/subject_images/subject_images_organised'
    sub_dirs = os.listdir(root_dir)
    #loop to go through all the subdirectories in the source directory
    
    if (peak_2):
        sub_dirs.remove('1')
        sub_dirs.remove('2')
        sub_dirs.remove('3')

    file = open("annotations_filtered_peak_two_partA.txt", "w")

    for sub_dir in sub_dirs:
        if sub_dir.endswith('.txt'):
            continue
        if sub_dir == 'physio':
            continue    
        sub_dir_path = os.path.join(root_dir, sub_dir)
        videos_list = os.listdir(sub_dir_path)
        
        count_missing_videos = 0
        count_videos_under_70 = 0
        count_videos_under_50 = 0
        count_videos_under_30 = 0
        
        
        for file_dir in videos_list:
            sub_video_path = os.path.join(sub_dir_path, file_dir)
            #count nyumber of images in each folder
            count = 0
            for sub_video in os.listdir(sub_video_path):
                count = count + 1
            
            if count > 70: 
                if sub_dir == '0' or sub_dir == '1' or sub_dir == '2' or sub_dir == '3' or sub_dir == '4':    
                    # write_file = os.path.join(sub_dir, file_dir) + " " + '1' + " " +str(count) + " " + str(sub_dir)
                    if sub_dir == '0':
                        class_label = '0'
                    elif sub_dir == '1':
                        class_label = '1'
                    elif sub_dir == '2':
                        class_label = '2'
                    elif sub_dir == '3':
                        class_label = '3'
                    elif sub_dir == '4':
                        class_label = '1' if (peak_2) else '4'
                    write_file = os.path.join(sub_dir, file_dir) + " " + '50' + " " + "70" + " " + class_label # only peak
                    file.write(write_file + "\n")
            
            else: 
                if (count <= 70): count_videos_under_70 += 1
                if (count <= 50): count_videos_under_50 += 1
                if (count < 30): count_videos_under_30 += 1     
                print(f'count {count} for {sub_video_path}')
            
        print(f'number of video with frames under 70 : {count_videos_under_70}')
        print(f'number of video with frames under 50 : {count_videos_under_50}')
        print(f'number of video with frames under 30 : {count_videos_under_30}')
        
    file.close()  

# create_annotxt()






# def count_labels(file_path):
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#         labels = [line.strip().split()[-1] for line in lines]

#     label_counts = {}
#     for label in labels:
#         if label in label_counts:
#             label_counts[label] += 1
#         else:
#             label_counts[label] = 1

#     return label_counts

# # File paths for train, validation, and test sets
# train_file = 'train.txt'
# val_file = 'val.txt'
# test_file = 'test.txt'

# # Count labels in each file
# train_label_counts = count_labels(train_file)
# val_label_counts = count_labels(val_file)
# test_label_counts = count_labels(test_file)

# # Print label counts
# print(f"Label counts in train set: {train_label_counts}")
# print(f"Label counts in validation set: {val_label_counts}")
# print(f"Label counts in test set: {test_label_counts}")



# code to write current file name and folder name to a text file
# import os
# import sys
#
# # Open a file
# path = "/var/www/html/"
# dirs = os.listdir( path )
#   
# # This would print all the files and directories
# for file in dirs:
#    print (file)
#    f = open("demofile2.txt", "a")
#    f.write(file + "\n")
#    f.close()


