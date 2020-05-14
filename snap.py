import cv2
import numpy as np
import imutils

from PIL import Image 
from tools.snap_grabcut import *
from tools.snap_pytorch import *

class Snap:

    def __init__ (self):
        self.GRABCUT = 4
        self.IMPROVED_GRABCUT = 8
        self.PYTORCH_MODEL = 10

    def snap_algorithm(self, *args):

        # Grabcut Implementation
        if (args[0] == 4 or args[0] == 8) and len(args) == 6:
            if(isinstance(args[1], np.ndarray)):
                img = args[1]
            else:
                print("First argument should be an image")
                return ValueError
            if isinstance(args[2], (float, int)) and isinstance(args[3], (float, int)) and isinstance(args[4], (float, int)) and isinstance(args[5], (float, int)):
                x1 = int(args[2])
                y1 = int(args[3])
                x2 = int(args[4])
                y2 = int(args[5])
            else:
                print("Argument 3 to 6 should be int or float")
                return ValueError

            w = x2 - x1
            h = y2 - y1

            print("Using Improved Grabcut Algorithm")
            (H,W) = img.shape[:2]
            display = img.copy()

            # cropping and filtering
            img_crop = img[int(y1):int(y2), int(x1):int(x2)]
            img = cv2.medianBlur(img_crop, 5)

            mask, rect = create_mask(5, w, h)
            
            thresh = grabcut(img, mask, rect)
            cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            area_list = []
            rec_list = []

            for c in cnts:    
                (nX, nY, w, h) = cv2.boundingRect(c)
                cnts_area = cv2.contourArea(c)
                rec_list.append(c)
                area_list.append(cnts_area)

            if area_list == []:
                print("No contours found")
                return mask, thresh, 0, 0, 0, 0

            else: 
                nX, nY, w, h = cv2.boundingRect(rec_list[area_list.index(max(area_list))])
                print("startX: ",nX, " startY: ", nY, " w: ", w, " h: ", h)
                cv2.rectangle(display, (nX, nY), (nX + w, nY + h), (0, 0xFF, 0), 4)
                px1 = int(nX + x1)
                py1 = int(nY + y1)
                px2 = int(nX + w + x1)
                py2 = int(nY + h + y1)
                return mask, thresh, px1, px2, py1, py2

        # PyTorch Segmentation
        elif args[0] == 10 and len(args)==6:
            if(isinstance(args[1], np.ndarray)):
                img = args[1]
            else:
                print("First argument should be an image path")
                return ValueError
            if isinstance(args[2], (float, int)) and isinstance(args[3], (float, int)) and isinstance(args[4], (float, int)) and isinstance(args[5], (float, int)):
                x1 = int(args[2])
                y1 = int(args[3])
                x2 = int(args[4])
                y2 = int(args[5])
            else:
                print("Argument 3 to 6 should be int or float")
                return ValueError
            
            display = img.copy()
            img_crop = img[int(y1):int(y2), int(x1):int(x2)]
            (H, W) = display.shape[:2]
            img = Image.fromarray(img, 'RGB')
        
            rgb = segment(img)
            out = imutils.resize(rgb, height=H, width=W)
            gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
            gray_crop = gray[int(y1):int(y2), int(x1):int(x2)]
            cnts, hierarchy = cv2.findContours(gray_crop, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            area_list = []
            rec_list = []

            for c in cnts:    
                (nX, nY, w, h) = cv2.boundingRect(c)
                cnts_area = cv2.contourArea(c)
                rec_list.append(c)
                area_list.append(cnts_area)

            if area_list == []:
                print("No contours found")
                return display, out, gray_crop, img_crop, 0, 0, 0, 0

            else: 
                nX, nY, w, h = cv2.boundingRect(rec_list[area_list.index(max(area_list))])
                print("startX: ",nX, " startY: ", nY, " w: ", w, " h: ", h)
                cv2.rectangle(img_crop, (nX, nY), (nX + w, nY + h), (0, 0xFF, 0), 3)
                px1 = int(nX + x1)
                py1 = int(nY + y1)
                px2 = int(nX + w + x1)
                py2 = int(nY + h + y1)
                return display, out, gray_crop, img_crop, px1, px2, py1, py2

        else:
            print("Proper usage of the snap.py: snap_algorithm(flag, img, px1, py1, px2, py2)")
            print("INPUT ERROR")
