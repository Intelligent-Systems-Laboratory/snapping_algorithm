import cv2
import numpy as np
import imutils
import argparse

# import this class to use the functions

class Snap:
    def __init__ (self) :
        self.SNAP_THRESHOLD = 1
        self.SNAP_OPTICAL_FLOW = 2
        self.SNAP_BACKGROUND_SUBTRACTION = 3
        self.SNAP_GRABCUT = 4

    def show_flow_hsv(self, flow, show_style=1):
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), np.uint8)

        if show_style == 1:
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 1] = 255
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        elif show_style == 2:
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            hsv[..., 2] = 255

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr


    # snapping_algorithm can be called by the following:
    # snap_algorithm(flag, img, x1, y1, x2, y2) -- purely vision-based
    # snap_algorithm(flag, prev, img, x1, y1, x2, y2) -- optical flow implementation

    def snap_algorithm(self, *args):
        print(len(args))
        # vision-based implementation of snapping
        if args[0] == 1 and len(args) == 6:
            if(isinstance(args[1], np.ndarray)):
                img = args[1]
            else:
                print("First argument should be an image")
                return ValueError
            if isinstance(args[2], (float, int)) and isinstance(args[3], (float, int)) or isinstance(args[4], (float, int)) or isinstance(args[5], (float, int)):
                x1 = args[2]
                y1 = args[3]
                x2 = args[4]
                y2 = args[5]
            else:
                print("Argument 2 to 5 should be int or float")
                return ValueError

            crop_img = img[y1:y2, x1:x2]
            cv2.imshow('Cropped Image', crop_img)
            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 75, 0xFF,
                                    cv2.THRESH_BINARY_INV)[1]
            # thresh = cv2.adaptiveThreshold(gray, 255,
            #                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            #                           cv2.THRESH_BINARY_INV, 11, 2)
            cv2.imshow('Threshold', thresh)
            cnts, hierarchy = cv2.findContours(
                thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                (nx1, ny1, w, h) = cv2.boundingRect(c)
                cnts_area = cv2.contourArea(c)
                CONTOUR_AREA_EVAL = 0.5 * (x2 - x1) * (y2 - y1)
                if cnts_area > CONTOUR_AREA_EVAL:
                    cv2.rectangle(crop_img, (nx1, ny1),
                             (nx1 + w, ny1 + h), (0, 0xFF, 0), 4)
                    print('new x1: ', nx1, 'new y1: ' , ny1, 'new x2: ', nx1 + w, 'new y2: ', ny1 + h)
                    cv2.imshow('crop', crop_img)    
                    return x1 + nx1, y1 + ny1, x2 + nx1 + w, y2 + ny1 + h
            print("No contours found")
            return 0, 0, 0 ,0

        # optical flow implementation of snapping
        elif args[0] == 2 and len(args) == 7:
            print("Using Optical Flow Method")
            if(isinstance(args[1], np.ndarray)):
                prev = args[1]
            else:
                print("First argument should be an image")
                return ValueError
            if(isinstance(args[2], np.ndarray)):
                img = args[2]
            else:
                print("Second argument should be an image")
                return ValueError
            if isinstance(args[3], (float, int)) and isinstance(args[4], (float, int)) or isinstance(args[5], (float, int)) or isinstance(args[6], (float, int)):
                x1 = args[3]
                y1 = args[4]
                x2 = args[5]
                y2 = args[6]
            else:
                print("Argument 3 to 6 should be int or float")
                return ValueError

            inst = cv2.DISOpticalFlow_create(cv2.DISOpticalFlow_PRESET_ULTRAFAST)
            inst.setUseSpatialPropagation(True)

            prev_crop = prev[y1:y2, x1:x2]
            crop_img = img[y1:y2, x1:x2]

            prevgray = cv2.cvtColor(prev_crop, cv2.COLOR_BGR2GRAY)

            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prevgray,gray,None,0.5,5,15,3,5,1.1,cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
            # flow = inst.calc(prevgray, gray, None)
            flow = self.show_flow_hsv(flow, show_style=1)
            cv2.imshow('flow', flow)
            cv2.imshow('prev', prev_crop)

            gray1 = cv2.cvtColor(flow, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray1, 45, 0xFF,
                                    cv2.THRESH_BINARY)[1]
            cv2.imshow('thresh', thresh)
            cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            area_list = []
            rec_list = []

            for c in cnts:
                (nX, nY, w, h) = cv2.boundingRect(c)
                cnts_area = cv2.contourArea(c)
                rec_list.append(nX, nY, w, h)
                area_list.append(cnts_area)

            nX, nY, w, h = rec_list[area_list.index(max(area_list))]
            print("startX: ",nX, " startY: ", nY, " w: ", w, " h: ", h)
            cv2.rectangle(crop_img, (nX, nY), (nX + w, nY + h), (0, 0xFF, 0), 4)
            cv2.imshow('crop', crop_img)
            return x1 + nX, y1 + nY, x2 + nX + w, y2 + nY + h

        # snapping with background subtraction implementation
        elif args[0] == 3 and len(args) == 7:
            print("Using Background Subtraction Method")
            if(isinstance(args[1], cv2.VideoCapture)):
                vid = args[1]
            else:
                print("First argument should be a video")
                return ValueError
            if(isinstance(args[2], int)):
                frame_no = args[2]
            else:
                print("Second argument should be a frame number")
                return ValueError
            if isinstance(args[3], (float, int)) and isinstance(args[4], (float, int)) or isinstance(args[5], (float, int)) or isinstance(args[6], (float, int)):
                x1 = args[3]
                y1 = args[4]
                x2 = args[5]
                y2 = args[6]
            else:
                print("Argument 3 to 6 should be int or float")
                return ValueError
            
            fgbg = cv2.createBackgroundSubtractorKNN()
            fgbg.setShadowValue(0)
            
            if frame_no > 50:
                vid.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 50)
            else:
                vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            while True:
                ret, frame = vid.read()
                fgmask = fgbg.apply(frame)
                if int(vid.get(cv2.CAP_PROP_POS_FRAMES)) > frame_no:
                    frame_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    fgmask_crop = fgmask[int(y1):int(y2), int(x1):int(x2)]
                    thresh = cv2.threshold(fgmask_crop, 20, 0xFF,
                                            cv2.THRESH_BINARY)[1]
                    cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    area_list = []
                    rec_list = []

                    for c in cnts:
                        bbox_list = []
                        (nX, nY, w, h) = cv2.boundingRect(c)
                        cnts_area = cv2.contourArea(c)

                        bbox_list.append(nX)
                        bbox_list.append(nY)
                        bbox_list.append(w)
                        bbox_list.append(h)

                        rec_list.append(bbox_list)
                        area_list.append(cnts_area)

                    if area_list == []:
                        print("No contours found")
                        return 0, 0, 0, 0

                    else:    
                        bbox_list = rec_list[area_list.index(max(area_list))]
                        nX = bbox_list[0]
                        nY = bbox_list[1]
                        w = bbox_list[2]
                        h = bbox_list[3]
                        print("startX: ",nX, " startY: ", nY, " w: ", w, " h: ", h)
                        px1 = int(x1 + nX)
                        py1 = int(y1 + nY)
                        px2 = int(x1 + nX + w)
                        py2 = int(y1 + nY + h)
                        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0xFF, 0), 4)
                        return thresh, fgmask_crop, frame_crop, frame

        # Grabcut Implementation
        elif args[0] == 4 and len(args) == 6:
            print("Using Grabcut Algorithm")
            if(isinstance(args[1], np.ndarray)):
                img = args[1]
            else:
                print("First argument should be an image")
                return ValueError
            if isinstance(args[2], (float, int)) and isinstance(args[3], (float, int)) or isinstance(args[4], (float, int)) or isinstance(args[5], (float, int)):
                x1 = args[2]
                y1 = args[3]
                x2 = args[4]
                y2 = args[5]
            else:
                print("Argument 2 to 5 should be int or float")
                return ValueError

            mask = np.zeros(img.shape[:2],np.uint8)
            bgdModel = np.zeros((1,65),np.float64)
            fgdModel = np.zeros((1,65),np.float64)

            rect = ((x1, y1), (x2, y2))
            print(type(rect))

            cv2.grabCut(img,mask,rect,bgdModel,fgdModel,10,cv2.GC_INIT_WITH_RECT) 
            display = img.copy()
            mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
            img = img*mask2[:,:,np.newaxis]
            img2 = np.where(img!=0,255,img).astype('uint8')
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) 
            img3 = cv2.threshold(img2,127,255,cv2.THRESH_BINARY)
            cnts, hierarchy = cv2.findContours(img2.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            area_list = []
            rec_list = []

            for c in cnts:    
                (nX, nY, w, h) = cv2.boundingRect(c)
                cnts_area = cv2.contourArea(c)
                rec_list.append(c)
                area_list.append(cnts_area)

            nX, nY, w, h = cv2.boundingRect(rec_list[area_list.index(max(area_list))])
            print("startX: ",nX, " startY: ", nY, " w: ", w, " h: ", h)
            cv2.rectangle(display, (nX, nY), (nX + w, nY + h), (0, 0xFF, 0), 4)
            return display


        else:
            print("Error in the arguments")
            return ValueError


    def click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("x-coor: ", x, ", y-coor:", y)
