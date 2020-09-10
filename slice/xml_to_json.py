
import numpy as np
import cv2
import os
import shutil
import random
import xml.etree.cElementTree as ET
from datetime import date
import json
def classToNum(vehicle_type):
  if vehicle_type == 'car':
    return 0
  if vehicle_type == 'suv':
    return 1
  if vehicle_type == 'van':
    return 2
  if vehicle_type == 'taxi':
    return 3
  if vehicle_type == 'truck':
    return 4
  if vehicle_type == 'motorcycle':
    return 5
  if vehicle_type == 'bicycle':
    return 6
  if vehicle_type == 'tricycle':
    return 7
  if vehicle_type == 'jeep':
    return 8
  if vehicle_type == 'bus':
    return 9


os.environ['DISPLAY'] = ':0'
#----cut video to frames----
#vidcap = cv2.VideoCapture('slice/slice_vids/Vid.mp4')
#success,image = vidcap.read()
#count = 0
#path = 'slice/slice_frames'
#while success:
#  filename = f'frame_{str(count).zfill(6)}'+'.jpg'
#  if count%10==0:
#    cv2.imwrite(os.path.join(path,filename), image)     # save frame as JPEG file      
#  success,image = vidcap.read()
#  print('Read a new frame: ', success)
#  count += 1
#----paths----
shutil.rmtree('slice/sliced')
os.mkdir('slice/sliced')
path_image = 'slice/slice_frames'
path_xml = 'slice/slice_xml'
path_out_train = 'slice/sliced/train'
path_out_valid = 'slice/sliced/valid'
if os.path.exists(path_out_train) != True:
  os.mkdir(path_out_train) 
if os.path.exists(path_out_valid) != True:
  os.mkdir(path_out_valid) 
path_out_json = 'slice/sliced/annotations'
if os.path.exists(path_out_json) != True:
  os.mkdir(path_out_json)

boxes = []
step = 30
#----define ROI for cropping----
imgvis = cv2.imread(os.path.join(path_image,'frame_000000.jpg'))
(height,width) = imgvis.shape[:2]
rect = cv2.selectROI('Select Area',imgvis,True)
cv2.destroyWindow('Select Area')
print(rect)
#rect = (201, 184, 1522, 698)
#----create labels for outputted crops----
k= 0  #iterator for step
folder_num = 0 #iterator for frame and xml folder
xml_list =os.listdir(path_xml)
for xml in range(int(len(xml_list))):        #iterate through all xml files in xml path
  tree = ET.ElementTree(file=os.path.join(path_xml,str('annotations'+str(xml)+'.xml')))
  root = tree.getroot()
  true = root.findall('image')
  for image in true:
    image_attrib = image.getchildren()
    W = int(image.get('width'))
    H = int(image.get('height'))
    for box in image_attrib:
      box_list = []
      box_atrib=box.getchildren()
      for atrib in box_atrib:
        if atrib.attrib['name'] == 'type':
          if atrib.text != 'ignore':
            x =float(box.get('xtl'))
            y =float(box.get('ytl'))
            w =float(box.get('xbr'))-float(box.get('xtl'))
            h =float(box.get('ybr'))-float(box.get('ytl'))
            box_type = atrib.text
            if int((x))>rect[0] and int((x+w))<rect[2] and int((y))>rect[1] and int((y+h))<rect[3] and k%step==0:
              box_list.append(image.attrib['name'])
              box_list.append(classToNum(box_type))
              box_list.append(x)
              box_list.append(y)
              box_list.append(w)
              box_list.append(h)
              box_list.append(folder_num)
              boxes.append(box_list)
              print(box_list[0])
              
    k+=1          
  folder_num+=1    
print(len(boxes))
boxes_valid=random.sample(boxes, int(max(20,len(boxes)*0.1)))
for box in boxes_valid:
  boxes.remove(box)
print('done')
print(len(boxes))
print(len(boxes_valid))

j=0
images = []
annotations = []
#create train image and annotations list of dicts
for box_i in range(len(boxes)):
  img = np.array(cv2.imread(os.path.join(f'{path_image}/{str(boxes[box_i][6])}',boxes[box_i][0]+'.jpg')),dtype=np.uint8)
  (H,W)=img.shape[:2]
  x_max = W
  y_max = H
  x_min = 0
  y_min = 0
  x=int(boxes[box_i][2])
  y=int(boxes[box_i][3])
  w=int(boxes[box_i][4])
  h=int(boxes[box_i][5])
  pad_h = int(0.3*w)
  pad_w = int(0.3*h)
  #if box_i%step == 0:
  for index in range(1,2):
    #write cropped image
    filename = str(j).zfill(6)
    img_crop = np.array(img[max(y_min,int(y-(pad_h*index)/2)):min(y_max,int(y+h+(pad_h*index)/2)), max(int(x-(pad_w*index)/2),x_min):min(x_max,int(x+w+(pad_w*index)/2))],dtype = np.uint8).copy()
    cv2.imwrite(os.path.join(path_out_train,filename+'.jpg'), img_crop)
    #update list of dict
    annotations.append({"id":j,"image_id":j,"category_id":(boxes[box_i][1]+1),"iscrowd":0,"area":(w*h),"bbox":[pad_w,pad_h,w,h],
    "segmentation":[[pad_w,
    pad_h,
    pad_w+w,
    pad_h,
    pad_w+w,
    pad_h+h,
    pad_w,
    pad_h+h]]})
    images.append({"id":j,"file_name":filename+".jpg","width":img_crop.shape[0],"height":img_crop.shape[1],"date_captured":str(date.today()),"license":1,"coco_url":"","fickr_url":""})
    #update image counter
    j+=1
    #write flipped image
    filename = str(j).zfill(6)
    cv2.imwrite(os.path.join(path_out_train,filename+'.jpg'), np.fliplr(img_crop))
    # save text here
    annotations.append({"id":j,"image_id":j,"category_id":boxes[box_i][1]+1,"iscrowd":0,"area":w*h,"bbox":[pad_w,pad_h,w,h],
    "segmentation":[[pad_w,
    pad_h,
    pad_w+w,
    pad_h,
    pad_w+w,
    pad_h+h,
    pad_w,
    pad_h+h]]})
    images.append({"id":j,"file_name":filename+".jpg","width":img_crop.shape[0],"height":img_crop.shape[1],"date_captured":str(date.today()),"license":1,"coco_url":"","fickr_url":""})
    
    print(str(j) + ' ' + str(boxes[box_i][6]) + ' ' + str(boxes[box_i][0]))

    #update image counter
    j+=1
  #cv2.rectangle(img, (int(x-w/2),int(y-h/2)), (int(x+w/2),int(y+h/2)), (0xFF, 0xFF, 0), 4)
  #cv2.imshow(boxes[box_i][0],img)
  #cv2.rectangle(img_crop, ((int((w+pad_w)/2)-int(w/2)),(int((h+pad_h)/2)-int(h/2))), ((int((w+pad_w)/2)+int(w/2)),(int((h+pad_h)/2)+int(h/2))), (0xFF, 0xFF, 0), 4)
  #cv2.imshow(boxes[box_i][0],img_crop)
  #cv2.waitKey(300)
  #cv2.destroyAllWindows()
  #break
  #cv2.imshow(boxes[box_i][0],img_crop)
  #cv2.waitKey(10000)
  #cv2.destroyAllWindows()
i=0
images_valid = []
annotations_valid = []
#create valid image and annotations list of dicts
for box_i in range(len(boxes_valid)):
  img = np.array(cv2.imread(os.path.join(f'{path_image}/{str(boxes_valid[box_i][6])}',boxes_valid[box_i][0]+'.jpg')),dtype=np.uint8)
  (H,W)=img.shape[:2]
  x_max = W
  y_max = H
  x_min = 0
  y_min = 0
  x=int(boxes_valid[box_i][2])
  y=int(boxes_valid[box_i][3])
  w=int(boxes_valid[box_i][4])
  h=int(boxes_valid[box_i][5])
  pad_h = int(0.3*w)
  pad_w = int(0.3*h)
  #if box_i%step == 0:
  for index in range(1,4):
    #write cropped image
    filename = str(i).zfill(6)
    img_crop = np.array(img[max(y_min,int(y-(pad_h*index)/2)):min(y_max,int(y+h+(pad_h*index)/2)), max(int(x-(pad_w*index)/2),x_min):min(x_max,int(x+w+(pad_w*index)/2))],dtype = np.uint8).copy()
    cv2.imwrite(os.path.join(path_out_valid,filename+'.jpg'), img_crop)
    #update list of dict
    annotations_valid.append({"id":i,"image_id":i,"category_id":(boxes_valid[box_i][1]+1),"iscrowd":0,"area":(w*h),"bbox":[pad_w,pad_h,w,h],
    "segmentation":[[pad_w,
    pad_h,
    pad_w+w,
    pad_h,
    pad_w+w,
    pad_h+h,
    pad_w,
    pad_h+h]]})
    images_valid.append({"id":i,"file_name":filename + ".jpg","width":img_crop.shape[0],"height":img_crop.shape[1],"date_captured":str(date.today()),"license":1,"coco_url":"","fickr_url":""})
    #update image counter
    i+=1
    #write flipped image
    filename = str(i).zfill(6)
    cv2.imwrite(os.path.join(path_out_valid,filename+'.jpg'), np.fliplr(img_crop))
    # save text here
    annotations_valid.append({"id":i,"image_id":i,"category_id":boxes_valid[box_i][1]+1,"iscrowd":0,"area":w*h,"bbox":[pad_w,pad_h,w,h],
    "segmentation":[[pad_w,
    pad_h,
    pad_w+w,
    pad_h,
    pad_w+w,
    pad_h+h,
    pad_w,
    pad_h+h]]})
    images_valid.append({"id":i,"file_name":filename+".jpg","width":img_crop.shape[0],"height":img_crop.shape[1],"date_captured":str(date.today()),"license":1,"coco_url":"","fickr_url":""})
    
    print(str(i) + ' ' + str(boxes_valid[box_i][6]) + ' ' + str(boxes_valid[box_i][0]))

    #update image counter
    i+=1
  #cv2.rectangle(img, (int(x-w/2),int(y-h/2)), (int(x+w/2),int(y+h/2)), (0xFF, 0xFF, 0), 4)
  #cv2.imshow(boxes[box_i][0],img)
  #cv2.rectangle(img_crop, ((int((w+pad_w)/2)-int(w/2)),(int((h+pad_h)/2)-int(h/2))), ((int((w+pad_w)/2)+int(w/2)),(int((h+pad_h)/2)+int(h/2))), (0xFF, 0xFF, 0), 4)
  #cv2.imshow(boxes[box_i][0],img_crop)
  #cv2.waitKey(300)
  #cv2.destroyAllWindows()
  #break
  #cv2.imshow(boxes[box_i][0],img_crop)
  #cv2.waitKey(10000)
  #cv2.destroyAllWindows()
print('hello baby')
#create valid json
filename_valid = 'instances_valid'
json_out = {'info':{'description':"",'url':"",'version':"",'year':int(date.today().year)},
'licenses':[{'id':1,'name':"",'url':""}],
'categories':[{'id':1,'name':"car",'supercategory':"None"},
{'id':2,'name':"suv",'supercategory':"None"},{'id':3,'name':"van",'supercategory':"None"},
{'id':4,'name':"taxi",'supercategory':"None"},{'id':5,'name':"truck",'supercategory':"None"},
{'id':6,'name':"motorcycle",'supercategory':"None"},{'id':7,'name':"bicycle",'supercategory':"None"},
{'id':8,'name':"tricycle",'supercategory':"None"},{'id':9,'name':"jeep",'supercategory':"None"},
{'id':10,'name':"bus",'supercategory':"None"}],
'images':images_valid,
'annotations':annotations_valid}
print('finished')
with open(os.path.join(path_out_json,filename_valid+'.txt'),'w') as outfile:
  json.dump(json_out,outfile)
#create train json
filename_train = 'instances_train'

json_out = {'info':{'description':"",'url':"",'version':"",'year':int(date.today().year)},
'licenses':[{'id':1,'name':"",'url':""}],
'categories':[{'id':1,'name':"car",'supercategory':"None"},
{'id':2,'name':"suv",'supercategory':"None"},{'id':3,'name':"van",'supercategory':"None"},
{'id':4,'name':"taxi",'supercategory':"None"},{'id':5,'name':"truck",'supercategory':"None"},
{'id':6,'name':"motorcycle",'supercategory':"None"},{'id':7,'name':"bicycle",'supercategory':"None"},
{'id':8,'name':"tricycle",'supercategory':"None"},{'id':9,'name':"jeep",'supercategory':"None"},
{'id':10,'name':"bus",'supercategory':"None"}],
'images':images,
'annotations':annotations}
print('finished')
with open(os.path.join(path_out_json,filename_train+'.txt'),'w') as outfile:
  json.dump(json_out,outfile)
os.rename(os.path.join(path_out_json,filename_train+'.txt'),os.path.join(path_out_json,filename_train+'.json'))
os.rename(os.path.join(path_out_json,filename_valid+'.txt'),os.path.join(path_out_json,filename_valid+'.json'))
