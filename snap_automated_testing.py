import cv2
import numpy as np
import imutils
import argparse
import os.path

from os import path

from snap import Snap

def run(vid_name,FRAME_NO,bboxx1,bboxy1,bboxx2,bboxy2,folder_name,snap_type):
    # the video could only use up to 400 frames for high accuracy
    vid = cv2.VideoCapture(vid_name)
    
    # THE PARAMETERS TO CHANGE
    # Coordinates of the resized image, not the full size
    vid.set(1, FRAME_NO)
    ret, img = vid.read()
    (H,W) = img.shape[:2]
    imgvis = imutils.resize(img, width=1028)
    ratioW = W/1028
    thresh, frame_crop, fgmask_crop, imgvis = snap.snap_algorithm(snap.SNAP_BACKGROUND_SUBTRACTION, vid, FRAME_NO, bboxx1*ratioW, bboxy1*ratioW, bboxx2*ratioW, bboxy2*ratioW)

    imgvis = imutils.resize(imgvis, width = 1028)
    if snap_type == snap.SNAP_BACKGROUND_SUBTRACTION:
        cv2.imwrite(folder_name+'/frame_'+str(frame_no)+'_thresh.jpg', thresh)
        cv2.imwrite(folder_name+'/frame_'+str(frame_no)+'_crop.jpg', frame_crop)
        cv2.imwrite(folder_name+'/frame_'+str(frame_no)+'_fgmask_crop.jpg', fgmask_crop)
        cv2.imwrite(folder_name+'/frame_'+str(FRAME_NO)+'.jpg',imgvis)

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--run_from_file", required=False, default=None, help="process coordinates from file")
args = vars(ap.parse_args())

snap = Snap()
snap_type = snap.SNAP_BACKGROUND_SUBTRACTION
i = 1

if args["run_from_file"] == None:
    args["run_from_file"] = ["small_boxes.txt", "medium_boxes.txt", "large_boxes.txt"]
else:
    args["run_from_file"] = [args["run_from_file"]]
    
for filename in args["run_from_file"]:
    if path.exists(filename[0:-4]) == False:
        os.mkdir(filename[0:-4])
        print(str(filename[0:-4]) + " directory made")
    with open (filename, 'rt') as myfile:
        for line in myfile:
            if i%8 == 1:
                video = line.rstrip('\n').replace('vid name: ','')
                folder_name = str(filename[0:-4]) + "/" + str(video[7:-4])
                print(folder_name)
                print('folder name: ' + folder_name)
                if path.exists(folder_name) == False:
                    os.mkdir(folder_name)
                    print(str(folder_name) + " directory made")
            if i%8 == 2:
                frame_no = int(line.rstrip('\n').replace('frame no: ',''))
            if i%8 == 3:
                x1 = int(line.rstrip('\n').replace('x1: ',''))
            if i%8 == 4:
                y1 = int(line.rstrip('\n').replace('y1: ',''))
            if i%8 == 5:
                x2 = int(line.rstrip('\n').replace('x2: ',''))
            if i%8 == 6:
                y2 = int(line.rstrip('\n').replace('y2: ',''))
            if i%8 == 7:
                print(video)
                print(frame_no)
                print(int(x1))
                print(int(y1))
                print(int(x2))
                print(int(y2))
            if i%8 == 0:
                run(video,frame_no,x1,y1,x2,y2,folder_name,snap_type)
                print("\n\n")
            i = i+1