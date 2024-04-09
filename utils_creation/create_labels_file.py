import cv2
import sys
import os
import csv
import pandas as pd

from PIL import Image

#/home/gpa/Documents/Work/Data/RehabData/RehabData

def main():
    # col_list = ['subject_name', 'class_id', 'sample_name']
    # df = pd.read_csv('starting_point/samples.csv', sep='\t', usecols=col_list)
    file = open("sub_all_labels.txt", "w")
    # root_dir = 'sub_img_red_classes'
    root_dir = '/projets2/AS84330/Datasets/Biovid/PartA/subject_images'
    
    for sub_dir in os.listdir(root_dir):
        sub_dir_path = os.path.join(root_dir, sub_dir)

        videos_list = os.listdir(sub_dir_path)
        # exclude_point = "101809_m_59"
        # videos_list = [video for video in videos_list if video != exclude_point]

		# filter out "PA1" and "PA2" videos 
		# videos_list = [video for video in videos_list if "PA1" not in video and "PA2" not in video]
        for file_dir in videos_list:
            sub_video_path = os.path.join(sub_dir_path, file_dir)
            print(file_dir)
            vid_label = 0
            if "PA1" in file_dir :
                # continue
                vid_label = 1
            elif "PA2" in file_dir :
                # continue
                vid_label = 2
            elif "PA3" in file_dir :
                # continue
                vid_label = 3
            elif "PA4" in file_dir :
                vid_label = 4
            for sub_video in os.listdir(sub_video_path):
                write_file = os.path.join(sub_video_path, sub_video) + " " + str(vid_label)
                print(write_file)
                file.write(write_file + "\n")

            # print(label)
            # print(os.listdir(img_path))

	# for i in range(len(df['subject_name'])):
	# 	
	# file.close()


def check():
    #check number of images in each folder
    root_dir = 'biovid_classes'
    lst= []
    miss_vid_list = []
    for sub_dir in os.listdir(root_dir):
        sub_dir_path = os.path.join(root_dir, sub_dir)
        count = 0
        if sub_dir.endswith(".txt"):
            continue
        
        #calculate number of images in each folder
        for file_dir in os.listdir(sub_dir_path):
            frame_count = 0
            sub_video_path = os.path.join(sub_dir_path, file_dir)
            # print(file_dir)
            # print(len(os.listdir(sub_video_path)))
            count += len(os.listdir(sub_video_path))
            frame_count += len(os.listdir(sub_video_path))
            if (frame_count != 75):
                print(frame_count)
                print(sub_video_path)
                miss_vid_list.append(file_dir)
        if (count != 7500):
            print(count)
            # print(sub_dir)
            lst.append(str(sub_dir))
    # print(lst)
    print(len(lst))
    # print(lst)
    print(miss_vid_list)
    len(miss_vid_list)
    # subtract lst from os.listdir(root_dir) to get the missing folders 
    x=set(os.listdir(root_dir)) - set(lst)
    # print(x)
    # print(len(x))
    print(len(miss_vid_list))




if __name__ == "__main__":
	main()
    # check()
