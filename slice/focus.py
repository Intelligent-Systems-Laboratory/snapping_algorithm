import numpy as np
import cv2
import os
import random
os.environ['DISPLAY'] = ':0'

path_image = 'slice/slice_frames'
path_labels = 'slice/slice_labels'
path_out_image = 'slice/sliced/images'
path_out_txt = 'slice/sliced/labels'
path_out_train = 'slice'
boxes = []
imgvis = cv2.imread(os.path.join(path_image,'frame_000000.jpg'))
(height,width) = imgvis.shape[:2]
rect = cv2.selectROI('Select Area',imgvis,True)
cv2.destroyWindow('Select Area')
print(rect)

for im_number in range(len(os.listdir(path_image))):
  #print(f'frame_{str(im_number).zfill(6)}')
  filename = f'frame_{str(im_number).zfill(6)}'
  my_file = open(path_labels+'/'+filename+'.txt')
  box_lines = my_file.readlines()
  my_file.close()

  line_no = 0
  for line in box_lines:
    box_attributes = line.split()
    box_attributes = [filename,box_attributes[0],[float(box_attributes[1]),float(box_attributes[2]),float(box_attributes[3]),float(box_attributes[4])],line_no]
    if box_attributes[2][0]*width>=rect[0] and box_attributes[2][0]*width<=(rect[0]+rect[2]) and box_attributes[2][1]*height>=rect[1] and box_attributes[2][1]*height<=(rect[1]+rect[3]):
      boxes.append(box_attributes)
    line_no+=1
  box_lines = []


print(len(boxes))

i=0
train_list=[]
for box in boxes:
  pad_h = int(np.random.rand()*100)
  pad_w = int(np.random.rand()*100)
  img = np.array(cv2.imread(os.path.join(path_image,box[0]+'.jpg')),dtype=np.uint8)
  print(str(box[0]) +' '+ str(box[3]))
  (H,W)=img.shape[:2]
  x=int(box[2][0]*W)
  y=int(box[2][1]*H)
  w=int(box[2][2]*W)
  h=int(box[2][3]*H)
  img_crop = np.array(img[int(y-(h+pad_h)/2):int(y+(h+pad_h)/2), int (x-(w+pad_w)/2):int(x+(w+pad_w)/2)],dtype = np.uint8).copy()
  filename_out = str(i).zfill(6)
  i+=1
  cv2.imwrite(os.path.join(path_out_image,filename_out+'.jpg'), img_crop)
  out_txt = open(os.path.join(path_out_txt,(filename_out+'.txt')),'w')
  out_txt.write(f'{box[1]} {(w+pad_w)/(2*(w+pad_w)):.6f} {(h+pad_h)/(2*(h+pad_h)):.6f} {(w)/(w+pad_w):.6f} {(h)/(h+pad_h):.6f}')
  out_txt.close()
  train_list.append(str("data/custom/images/"+filename_out+".jpg\n"))
  #cv2.rectangle(img, (int(x-w/2),int(y-h/2)), (int(x+w/2),int(y+h/2)), (0xFF, 0xFF, 0), 4)
  #cv2.imshow(box[0],img)
  #cv2.rectangle(img_crop, ((int((w+pad_w)/2)-int(w/2)),(int((h+pad_h)/2)-int(h/2))), ((int((w+pad_w)/2)+int(w/2)),(int((h+pad_h)/2)+int(h/2))), (0xFF, 0xFF, 0), 4)
  #cv2.imshow(box[0],img_crop)
  #cv2.waitKey()
  #cv2.destroyAllWindows()
  #break
  #cv2.imshow(box[0],img_crop)
  #cv2.waitKey(10000)
  #cv2.destroyAllWindows()
out_txt = open(os.path.join(path_out_train,('CVAT.txt')),'w')
for line in train_list:
  out_txt.write(line)
out_txt.close()
valid_list = random.sample(train_list,30)
out_valid = open(os.path.join(path_out_train,('CVAT_valid.txt')),'w')
for line in valid_list:
  out_valid.write(line)
out_txt.close()
  


