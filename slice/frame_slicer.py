import numpy as np
import cv2
#import imutils
import os
#path = 'C:\\\Users\\\Asus\\\Devcrap\\\snapping_algorithm\\\slice_frames\\\slice_vids\\\Vid.mp4'
#vid = cv2.VideoCapture(path)

#for frameNo in range(1500):
#    vid.set(1, frameNo-1)
#    ret, img = vid.read()
#    filename = f'frame_{str(frameNo).zfill(6)}'+'.jpg'
#    path = './slice_frames/frames'
#    cv2.imshow('frame',img)
#    cv2.imwrite(os.path.join(path,filename),img)
#    print(f'success: {filename}')
path_vid = 'slice/slice_vids'
path_frames = 'slice/slice_frames'
folder_list=sorted(os.listdir(path_vid))
#for folder in folder_list:
#  path_vid_current = os.path.join(path_vid,folder)
vid_list = sorted(os.listdir(path_vid))
i= 0
for vid in range(len(vid_list)):
  vidcap = cv2.VideoCapture(os.path.join(path_vid,f'{vid}.mp4'))
  success,image = vidcap.read()
  count = 0
  #path_frame = f'{path_frames}/{folder}'
  #if os.path.exists(path_frame) !=1:
  #  os.mkdir(path_frame)
  path_frame = f'{path_frames}/{i}'
  if os.path.exists(path_frame) !=1:
    os.mkdir(path_frame)
  while success:
    filename = f'frame_{str(count).zfill(6)}'+'.jpg'
    cv2.imwrite(os.path.join(path_frame,filename), image)     # save frame as JPEG file      
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
  print(f'{i} finished')
  i+=1
  