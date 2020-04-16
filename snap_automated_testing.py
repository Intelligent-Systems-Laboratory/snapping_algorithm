import cv2
import numpy as np
import imutils
import argparse
import os.path
import csv

from datetime import datetime
from os import path
from snap import Snap
from array import *

def time():
    dateTimeObj = datetime.now()
    return dateTimeObj


def time_print(dateTimeObj):
    print(dateTimeObj.year, '/', dateTimeObj.month, '/', dateTimeObj.day)
    print(dateTimeObj.hour, ':', dateTimeObj.minute, ':', dateTimeObj.second, '.', dateTimeObj.microsecond)

# Object definition of a test case
class Case(object):
    def __init__(self, videoFile, frame, x1, x2, y1, y2):
        self.videoFile = videoFile
        self.frame = frame
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def printCase(self):
        print("video: " + str(self.videoFile))
        print("frame: " + str(self.frame))
        print("x1: " + str(self.x1))
        print("y1: " + str(self.y1))
        print("x2: " + str(self.x2))
        print("y2: " + str(self.y2))


# Formats text file to an array for easier access
def textFileToCaseList(filename):
    caseList = []
    i = 1
    with open (filename, 'rt') as myfile:
        for line in myfile:
            if i%8 == 1:
                video = line.rstrip('\n').replace('vid name: ','')
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
                caseList.append(Case(video, frame_no, x1, x2, y1 ,y2))
                        
            i = i+1

    return caseList

# Measures accuracy by finding the ratio between the intersection and union of generated bounding box and the ground truth bounding box
def measureAccuracy (truth_xmin, truth_xmax, truth_ymin, truth_ymax, box_xmin, box_xmax, box_ymin, box_ymax):
    truthArea = (truth_xmax - truth_xmin) * (truth_ymax - truth_ymin)
    boxArea = (box_xmax - box_xmin) * (box_ymax - box_ymin)
    leftIntersect = max(truth_xmin, box_xmin)
    rightIntersect = min(truth_xmax, box_xmax)
    bottomIntersect = max(truth_ymin, box_ymin)
    topIntersect = min(truth_ymax, box_ymax)

    if (rightIntersect < leftIntersect) or (topIntersect < bottomIntersect):
        accuracy = 0
    else:
        overlapArea = (rightIntersect - leftIntersect) * (topIntersect - bottomIntersect)
        totalArea = truthArea + boxArea - overlapArea
        accuracy = overlapArea / totalArea

    return accuracy

def run(vid_name,FRAME_NO,bboxx1,bboxy1,bboxx2,bboxy2,folder_name,snap_type):
    # the video could only use up to 400 frames for high accuracy
    vid = cv2.VideoCapture(vid_name)
    
    # THE PARAMETERS TO CHANGE
    # Coordinates of the resized image, not the full size
    vid.set(1, FRAME_NO)
    ret, img = vid.read()
    (H, W) = img.shape[:2]
    imgvis = imutils.resize(img, width=1028)
    ratioW = W/1028

    # Return bbox coordinates along with generated images
    if snap_type == snap.SNAP_BACKGROUND_SUBTRACTION_KNN or snap_type == snap.SNAP_BACKGROUND_SUBTRACTION_MOG2 or snap_type == snap.SNAP_BACKGROUND_SUBTRACTION_CNT:
        thresh, fgmask_crop, frame_crop, imgvis, box_xmin, box_xmax, box_ymin, box_ymax = snap.snap_algorithm(snap_type, vid, FRAME_NO, bboxx1*ratioW, bboxy1*ratioW, bboxx2*ratioW, bboxy2*ratioW)
    elif snap_type == snap.SNAP_BACKGROUND_SUBTRACTION_CNT_NO_SHADOW:
        thresh, fgmask_crop, frame_crop, imgvis, box_xmin, box_xmax, box_ymin, box_ymax = snap.snap_algorithm(snap_type, vid, vid_name, FRAME_NO, bboxx1*ratioW, bboxy1*ratioW, bboxx2*ratioW, bboxy2*ratioW)
    elif snap_type == snap.SNAP_GRABCUT:
        thresh, grabcut_crop, display = snap.snap_algorithm(snap_type, imgvis, bboxx1*ratioW, bboxy1*ratioW, bboxx2*ratioW, bboxy2*ratioW)

    imgvis = imutils.resize(imgvis, width = 1028)
    if snap_type == snap.SNAP_BACKGROUND_SUBTRACTION_KNN or snap_type == snap.SNAP_BACKGROUND_SUBTRACTION_MOG2 or snap_type == snap.SNAP_BACKGROUND_SUBTRACTION_CNT or snap.SNAP_BACKGROUND_SUBTRACTION_CNT_NO_SHADOW:
        # cv2.imwrite(folder_name+'/frame_'+str(frame_no)+'_bgimg.jpg', background)
        cv2.imwrite(folder_name+'/frame_'+str(frame_no)+'_thresh.jpg', thresh)
        cv2.imwrite(folder_name+'/frame_'+str(frame_no)+'_crop.jpg', frame_crop)
        cv2.imwrite(folder_name+'/frame_'+str(frame_no)+'_fgmask_crop.jpg', fgmask_crop)
        cv2.imwrite(folder_name+'/frame_'+str(FRAME_NO)+'.jpg',imgvis)
    elif snap_type == snap.SNAP_GRABCUT:
        cv2.imwrite(folder_name+'frame_'+str(frame_no)+'_thresh_grabcut.jpg', thresh)
        cv2.imwrite(folder_name+'/frame_'+str(frame_no)+'_crop_grabcut.jpg', grabcut_crop)
        cv2.imwrite(folder_name+'/frame_'+str(FRAME_NO)+'_grabcut.jpg',display)

    # Downsize generated bounding box coordinates
    box_xmin /= ratioW
    box_xmax /= ratioW
    box_ymin /= ratioW
    box_ymax /= ratioW    

    return box_xmin, box_xmax, box_ymin, box_ymax


start_time = time()
ap = argparse.ArgumentParser()
ap.add_argument("-r", "--run_from_file", required=False, default=None, help="process coordinates from file")
args = vars(ap.parse_args())

snap = Snap()
snap_type = snap.SNAP_BACKGROUND_SUBTRACTION_CNT    # Choose snap algorithm here
i = 1

# Sets list of ground truth bounding boxes
groundTruthFileName = "ground_truth.txt"
groundTruthList = textFileToCaseList(groundTruthFileName)

if args["run_from_file"] == None:
    args["run_from_file"] = ["small_boxes.txt"]
else:
    args["run_from_file"] = [args["run_from_file"]]

for filename in args["run_from_file"]:

    if path.exists(filename[0:-4]) == False:
        os.mkdir(filename[0:-4])
        print(str(filename[0:-4]) + " directory made")

    smallBoxesList = textFileToCaseList(filename)
    for case in smallBoxesList:
        # Initialize properties of test case
        video = case.videoFile
        frame_no = case.frame
        x1 = case.x1
        y1 = case.y1
        x2 = case.x2
        y2 = case.y2
        folder_name = str(filename[0:-4]) + "/" + str(case.videoFile[7:-4])

        case.printCase()
        print(folder_name)

        if path.exists(folder_name) == False:
            os.mkdir(folder_name)
            print(str(folder_name) + " directory made")

        # Retrieve corresponding ground truth case
        index = smallBoxesList.index(case)
        truth = groundTruthList[index]
        truth_x1 = truth.x1
        truth_x2 = truth.x2
        truth_y1 = truth.y1
        truth_y2 = truth.y2

        # Run snapping algorithm
        box_xmin, box_xmax, box_ymin, box_ymax = run(video,frame_no,x1,y1,x2,y2,folder_name,snap_type)
        run(video,frame_no,x1,y1,x2,y2,folder_name,snap_type)
        
        # Measure accuracy of adjusted bounding box
        accuracy = measureAccuracy(truth_x1, truth_x2, truth_y1, truth_y2, box_xmin, box_xmax, box_ymin, box_ymax)
        print(str(accuracy*100) + '%')

        # Records accuracy measure in a csv file
        with open('accuracy.csv', 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([video, frame_no, accuracy])

        print("\n\n")

end_time = time()

time_print(start_time)
time_print(end_time)
