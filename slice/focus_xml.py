import numpy as np
import cv2
import os
import random
import xml.etree.cElementTree as ET
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
path_image = 'slice/slice_frames'
path_xml = 'slice/slice_xml'
path_out_image = 'slice/sliced/images'
if os.path.exists(path_out_image) != True:
  os.mkdir(path_out_image) 
path_out_txt = 'slice/sliced/labels'
if os.path.exists(path_out_txt) != True:
  os.mkdir(path_out_txt)
path_out_train = 'slice'
boxes = []
step = 5
#----define ROI for cropping----
#imgvis = cv2.imread(os.path.join(path_image,'frame_000000.jpg'))
#(height,width) = imgvis.shape[:2]
#rect = cv2.selectROI('Select Area',imgvis,True)
#cv2.destroyWindow('Select Area')
#print(rect)
rect = (201, 184, 1522, 698)
#----create labels for outputted crops----
j= 0  #iterator for step
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
            x =float(((float(box.get('xbr'))+float(box.get('xtl')))/(2*float(image.attrib['width']))))
            y =float((float(box.get('ybr'))+float(box.get('ytl')))/(2*float(image.attrib['height'])))
            w =float((float(box.get('xbr'))-float(box.get('xtl')))/float(image.attrib['width']))
            h =float((float(box.get('ybr'))-float(box.get('ytl')))/float(image.attrib['height']))
            box_type = atrib.text
            if int((x-w/2)*W)>rect[0] and int((x+w/2)*W)<rect[2] and int((y-h/2)*H)>rect[1] and int((y+h/2)*H)<rect[3]and j%step==0:
              box_list.append(image.attrib['name'])
              box_list.append(classToNum(box_type))
              box_list.append(x)
              box_list.append(y)
              box_list.append(w)
              box_list.append(h)
              box_list.append(folder_num)
              boxes.append(box_list)
              print(box_list[0])
    j=j+1          
  folder_num+=1    

print('done')
print(len(boxes))
i=0
train_list=[]

for box_i in range(len(boxes)):
  img = np.array(cv2.imread(os.path.join(f'{path_image}/{str(boxes[box_i][6])}',boxes[box_i][0]+'.jpg')),dtype=np.uint8)
  (H,W)=img.shape[:2]
  x_max = W
  y_max = H
  x_min = 0
  y_min = 0
  x=int(boxes[box_i][2]*W)
  y=int(boxes[box_i][3]*H)
  w=int(boxes[box_i][4]*W)
  h=int(boxes[box_i][5]*H)
  pad_h = int(0.3*w)
  pad_w = int(0.3*h)
  #if box_i%step == 0:
  for index in range(1,4):
    filename = str(i).zfill(6)
    img_crop = np.array(img[max(y_min,int(y-(h+pad_h*index)/2)):min(y_max,int(y+(h+pad_h*index)/2)), max(int(x-(w+pad_w*index)/2),x_min):min(x_max,int(x+(w+pad_w*index)/2))],dtype = np.uint8).copy()
    cv2.imwrite(os.path.join(path_out_image,filename+'.jpg'), img_crop)
    out_txt = open(os.path.join(path_out_txt,(filename+'.txt')),'w')
    out_txt.write(f'{boxes[box_i][1]} {(w+pad_w*index)/(2*(w+pad_w*index)):.6f} {(h+pad_h*index)/(2*(h+pad_h*index)):.6f} {(w)/(w+pad_w*index):.6f} {(h)/(h+pad_h*index):.6f}')
    out_txt.close()
    train_list.append(str("data/custom/images/"+filename+".jpg\n"))
    print(str(i) + ' ' + str(boxes[box_i][6]) + ' ' + str(boxes[box_i][0]))
    i+=1
    filename = str(i).zfill(6)
    cv2.imwrite(os.path.join(path_out_image,filename+'.jpg'), np.fliplr(img_crop))
    out_txt = open(os.path.join(path_out_txt,(filename+'.txt')),'w')
    out_txt.write(f'{boxes[box_i][1]} {(w+pad_w*index)/(2*(w+pad_w*index)):.6f} {(h+pad_h*index)/(2*(h+pad_h*index)):.6f} {(w)/(w+pad_w*index):.6f} {(h)/(h+pad_h*index):.6f}')
    out_txt.close()
    train_list.append(str("data/custom/images/"+filename+".jpg\n"))
    print(str(i) + ' ' + str(boxes[box_i][6]) + ' ' + str(boxes[box_i][0]))
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
out_txt = open(os.path.join(path_out_train,('CVAT.txt')),'w')
valid_list = random.sample(train_list,max(30,int(len(train_list)*0.01)))
for item in valid_list:
  train_list.remove(item)
for line in train_list:
  out_txt.write(line)
out_txt.close()
out_valid = open(os.path.join(path_out_train,('CVAT_valid.txt')),'w')
for line in valid_list:
  out_valid.write(line)
out_valid.close()
  


