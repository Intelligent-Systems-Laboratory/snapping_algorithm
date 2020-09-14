from tools.torchcut.models import *
from tools.torchcut.utils.datasets import *
from tools.torchcut.utils.utils import *
import cv2
import numpy
os.environ['DISPLAY'] = ':0'
def NumToClass(class_num):
  if class_num == 0:
    return 'car'
    
  if class_num == 1:
    return 'suv'
  if class_num == 2:
    return 'van'
  if class_num == 3:
    return 'taxi'
  if class_num == 4:
    return 'truck'
  if class_num == 5:
    return 'motorcycle'
  if class_num == 6:
    return 'bicycle'
  if class_num == 7:
    return 'tricycle'
  if class_num == 8:
    return 'jeep'
  if class_num == 9:
    return 'bus'
def snap_torchcut(img):
  CONFIG = '/home/janos/Desktop/Dev/snapping_algorithm/slice/yolov3-custom.cfg'
  WEIGHTS = '/home/janos/Desktop/Dev/snapping_algorithm/slice/best.pt'
  out = '/home/janos/Desktop/Dev/snapping_algorithm/slice/output'
  #imgsz = input_image.shape[:2]
  input_image = img
  imgsz = 416  
  device = torch_utils.select_device(device='0')
  #initialize model    
  model = Darknet(CONFIG, imgsz)
  #load .pt
  model.load_state_dict(torch.load(WEIGHTS, map_location=device)['model']) 
  model.to(device).eval()
  #img = cv2.imread('slice/slice_frames/5/frame_000000.jpg')
  #img = cv2.imread('/home/janos/Desktop/Dev/snapping_algorithm/1.jpg')
  #r = cv2.selectROI('Select Area',img)
  #cv2.destroyAllWindows()
  #rect = r
  # print(rect)
  #input_image = np.array(img[rect[1]:rect[3], rect[0]:rect[2]])
  #input_image = np.array(img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]])
  #cv2.imshow('OUT',input_image)
  #cv2.waitKey()
  
  # Padded resize
  img = letterbox(input_image, new_shape=416)[0]
  # Convert
  img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
  img = np.ascontiguousarray(img)
  #select device
  
  #initialize model
  #model = Darknet(CONFIG, imgsz)
  #convert image
  img = torch.from_numpy(img).to(device)
  img = img.float()
  img /= 255.0
  if img.ndimension() == 3:
      img = img.unsqueeze(0)
  #load .pt
  #model.load_state_dict(torch.load(WEIGHTS, map_location=device)['model']) 
  #model.to(device).eval()
  #pred = model(img,augment=False)[0]
  pred = model(img,augment=False)[0]
  #nonmaxsuppresion
  pred = non_max_suppression(pred)
  final_box = []
  for i, det in enumerate(pred):  # detections for image i
      if det is not None and len(det):
          # Rescale boxes from imgsz to im0 size
          det[:, :4] = scale_coords(img.shape[2:], det[:, :4], input_image.shape).round()
      final_box = det
      if final_box is not None:
        final_box = final_box.cpu().detach().numpy()[0]
        #cv2.rectangle(input_image,(final_box[0],final_box[1]),(final_box[2],final_box[3]),(0,0,255),2)
        #cv2.putText(input_image, f'{NumToClass(int(final_box[-1]))} {str(final_box[-2])}', (int(final_box[0]), int(final_box[1])-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
        #cv2.imshow('OUT',input_image)
        #print(NumToClass(int(final_box[-1]))+ ' '+ str(final_box[-2]))
        #cv2.waitKey()
        #cv2.destroyAllWindows()
      else:
        final_box = [0,0,0,0,0,0]
  return final_box