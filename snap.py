import cv2
from cv2 import bgsegm
import numpy as np
import imutils

class Snap:

    def __init__ (self):
        self.GRABCUT = 4
        self.IMPROVED_GRABCUT = 8

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

            bgdModel = np.zeros((1,65), np.float64)
            fgdModel = np.zeros((1,65), np.float64)
            w = x2 - x1
            h = y2 - y1
            
            if args[0] == 4:
                print("Using Grabcut Algorithm")
                mask = np.zeros(img.shape[:2],np.uint8)
                rect = (x1, y1, w, h)

            elif args[0] == 8: #Enhancement, placing filters and improving grabcut
                print("Using Improved Grabcut Algorithm")
                (H,W) = img.shape[:2]
                
                # cropping and filtering
                img_crop = img[int(y1):int(y2), int(x1):int(x2)]
                img = cv2.medianBlur(img_crop, 5)
                v_ex = 5

                # creating the mask
                mask = np.zeros(img.shape[:2],np.uint8)
                (mask_h, mask_w) = mask.shape[:2]

                for iter in range(0, mask_w-1):
                    mask[v_ex][iter] = 1
                    mask[v_ex+1][iter] = 1
                    mask[mask_h-v_ex][iter] = 1
                    mask[mask_h-v_ex-1][iter] = 1
                for iter in range(0, mask_h-1):
                    mask[iter][v_ex] = 1
                    mask[iter][v_ex+1] = 1
                    mask[iter][mask_w-v_ex] = 1
                    mask[iter][mask_w-v_ex-1] = 1
                rect = (v_ex,v_ex,w-v_ex,h-v_ex)

            cv2.grabCut(img,mask,rect,bgdModel,fgdModel,25,cv2.GC_BGD) 
            display = img.copy()
            mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
            grabcut_crop = img*mask2[:,:,np.newaxis]
            img2 = np.where(grabcut_crop!=0,255, grabcut_crop).astype('uint8')
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(img2,127,255,cv2.THRESH_BINARY)[1]
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
                return mask2, thresh, grabcut_crop, display, 0, 0, 0, 0

            else: 
                nX, nY, w, h = cv2.boundingRect(rec_list[area_list.index(max(area_list))])
                print("startX: ",nX, " startY: ", nY, " w: ", w, " h: ", h)
                cv2.rectangle(display, (nX, nY), (nX + w, nY + h), (0, 0xFF, 0), 4)
                px1 = int(nX + x1)
                py1 = int(nY + y1)
                px2 = int(nX + w + x1)
                py2 = int(nY + h + y1)
                return mask2, thresh, grabcut_crop, display, px1, px2, py1, py2

        else:
            print("Proper usage of the snap.py: snap_algorithm(flag, img, px1, py1, px2, py2)")
            print("INPUT ERROR")

            