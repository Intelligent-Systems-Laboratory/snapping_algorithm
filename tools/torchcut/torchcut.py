from tools.torchcut.models import *
from tools.torchcut.utils.datasets import *
from tools.torchcut.utils.utils import *
import cv2
import numpy
def snap_torchcut(img):
  # path to config file here
  CONFIG = './yolov3-custom.cfg'
  #path to weights here
  WEIGHTS = './Yolov3_best.pt'
  #imgsz = input_image.shape[:2]
  input_image = img
  imgsz = 416  
  device = torch_utils.select_device(device='0')
  #initialize model    
  model = Darknet(CONFIG, imgsz)
  #load .pt
  model.load_state_dict(torch.load(WEIGHTS, map_location=device)['model']) 
  model.to(device).eval()
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
      else:
        final_box = [0,0,0,0,0,0]
  return final_box
