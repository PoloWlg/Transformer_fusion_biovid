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



# Define the directory path
directory = "/projets/AS84330/Datasets/Biovid/PartA/biovid_classes_partA"

for sub_dir in os.listdir(directory):
    sub_dir_path = os.path.join(directory, sub_dir)
    videos_list = os.listdir(sub_dir_path)
    for file_dir in videos_list:
        sub_video_path = os.path.join(sub_dir_path, file_dir)
        image_files = [f for f in os.listdir(sub_video_path) if os.path.isfile(os.path.join(sub_video_path, f)) and f.lower().endswith(('.jpg', '.jpeg'))]
        image_files.sort()
        counter = 1
        num_digits = 5

        for image_file in image_files:
            # Generate the new file name with leading zeros
            new_name = f"img_{counter:0{num_digits}}.jpg"

            # Build the full paths
            old_path = os.path.join(sub_video_path, image_file)
            new_path = os.path.join(sub_video_path, new_name)

            # Rename the file
            os.rename(old_path, new_path)

            # Increment the counter
            counter += 1


# Get a list of all image files in the directory

# Sort the image files to ensure they are in order


# Define the counter variable
counter = 1

# Define the number of digits for the counter
num_digits = 5
print(image_files)
# Loop through the image files and rename them
for image_file in image_files:
    # Generate the new file name with leading zeros
    new_name = f"{counter:0{num_digits}}.png"

    # Build the full paths
    old_path = os.path.join(directory, image_file)
    new_path = os.path.join(directory, new_name)
    print('image_file: ', image_file)
    print('directory: ', directory)
    print('old_path: ', old_path)


    # Rename the file
    os.rename(old_path, new_path)

    # Increment the counter
    counter += 1

print("Image frames renamed successfully.")
