import cv2
from cv2 import bgsegm
from PIL import Image
import numpy as np
import imutils
import argparse

# import this class to use the functions

class Snap:
    def __init__ (self):
        self.SNAP_THRESHOLD = 1
        self.SNAP_OPTICAL_FLOW = 2
        self.SNAP_BACKGROUND_SUBTRACTION_KNN = 3
        self.SNAP_GRABCUT = 4
        self.SNAP_BACKGROUND_SUBTRACTION_MOG2 = 5
        self.SNAP_BACKGROUND_SUBTRACTION_CNT = 6
        self.SNAP_BACKGROUND_SUBTRACTION_CNT_NO_SHADOW = 7
        self.SNAP_IMPROVED_GRABCUT = 8
        self.SNAP_GRABCUT_WITH_CNT = 9
        self.SNAP_TO_BIGGER = 10
        self.previous_video = None
        self.previous_bgimage = None


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


    def set_previous_video(self, vid_name, bgimage):
        self.previous_video = vid_name
        self.previous_bgimage = bgimage
        # for mog2 usage, bgimage is fgbg storing the learned kernel


    def get_previous_bgimage(self):
        return self.previous_bgimage


    def is_same_video(self, vid_name):
        if vid_name == self.previous_video:
            return True
        else:
            return False


    def get_bgimage(self, vid, vid_name, frame_skip = 10, return_type = "image"):
        if self.is_same_video(vid_name) == False:
            if isinstance(vid, cv2.VideoCapture) == False:
                print("Video must be a cv2.VideoCapture")
                return None
            fgbg = bgsegm.createBackgroundSubtractorCNT()
            vid.set(cv2.CAP_PROP_POS_FRAMES, frame_skip - 1)
            bgimage_list = []
            if frame_skip == 0:
                vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                while True:
                    ret, frame = vid.read()
                    if ret:
                        fgmask = fgbg.apply(frame)
                    else:
                        break
                bgimage = fgbg.getBackgroundImage()
                self.set_previous_video(vid_name, bgimage)
            else: 
                vid.set(cv2.CAP_PROP_POS_FRAMES, frame_skip - 1)
                while True:
                    ret, frame = vid.read()
                    if ret:
                        fgmask = fgbg.apply(frame)
                    else:
                        break

                    if int(vid.get(cv2.CAP_PROP_POS_FRAMES)) % (frame_skip * 50) == 0:
                        bgimage_list.append(fgbg.getBackgroundImage())
                        
                    count = 0
                    for count in range(1,frame_skip):
                        if count == frame_skip:
                            break
                        else:
                            ret, frame = vid.read()
                self.set_previous_video(vid_name, bgimage_list)
            if return_type == "array":
                return bgimage_list
            elif return_type == "image" and frame_skip == 0:
                return bgimage
            elif return_type == "image":
                return bgimage_list[len(bgimage_list)-1]
            else:
                print("Valid return_type: array, image")
                return None

        else:
            print("Using previous data")
            bgimage_list = self.get_previous_bgimage()
            if return_type == "array":
                return bgimage_list
            elif return_type == "image" and frame_skip == 0:
                return bgimage_list
            elif return_type == "image":
                return bgimage_list[len(bgimage_list)-1]
            else:
                print("Valid return_type: array, image")
                return None


    def snap_algorithm(self, *args):
        # vision-based implementation of snapping
        if args[0] == 1 and len(args) == 6:
            if(isinstance(args[1], np.ndarray)):
                img = args[1]
            else:
                print("First argument should be an image")
                return ValueError
            if isinstance(args[2], (float, int)) and isinstance(args[3], (float, int)) or isinstance(args[4], (float, int)) or isinstance(args[5], (float, int)):
                x1 = int(args[2])
                y1 = int(args[3])
                x2 = int(args[4])
                y2 = int(args[5])
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
            return 0, 0, 0, 0


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


        # snapping with KNN background subtraction implementation
        elif args[0] == 3 and len(args) == 7:
            print("Using KNN Background Subtraction Method")
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
                        return thresh, fgmask_crop, frame_crop, frame

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
                        return thresh, fgmask_crop, frame_crop, frame, px1, px2, py1, py2


        # Grabcut Implementation
        elif (args[0] == 4 or args[0] == 8) and len(args) == 6:
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
                print("Using Enhanced Grabcut Algorithm")
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


        # MOG2 Background Subtraction
        elif args[0] == 5 and len(args) == 8:
            print("Using MOG2 Background Subtraction Method")
            if(isinstance(args[1], cv2.VideoCapture)):
                vid = args[1]
            else:
                print("First argument should be a video")
                return ValueError
            if(isinstance(args[2], str)):
                vid_name = args[2]
            else:
                print("Second argument should be a video name")
                return ValueError
            if(isinstance(args[3], int)):
                frame_no = args[3]
            else:
                print("Second argument should be a frame number")
                return ValueError
            if isinstance(args[4], (float, int)) and isinstance(args[5], (float, int)) or isinstance(args[6], (float, int)) or isinstance(args[7], (float, int)):
                x1 = args[4]
                y1 = args[5]
                x2 = args[6]
                y2 = args[7]
            else:
                print("Argument 3 to 6 should be int or float")
                return ValueError
            
            if self.is_same_video(vid_name) == False: 
                fgbg = cv2.createBackgroundSubtractorMOG2()
                fgbg.setShadowValue(0)

                while True:
                    ret, frame = vid.read()
                    if ret:
                        fgmask = fgbg.apply(frame)
                    else:
                        break
                self.set_previous_video(vid_name, fgbg)

            else:
                print("Using previous fgbg")
                fgbg = self.get_previous_bgimage()
                fgbg.setShadowValue(0)
                
            while True:
                vid.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)
                ret, frame = vid.read()
                fgmask = fgbg.apply(frame)
                frame_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                fgmask_crop = fgmask[int(y1):int(y2), int(x1):int(x2)]
                thresh = cv2.threshold(fgmask_crop, 25, 0xFF,
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
                    return thresh, fgmask_crop, frame_crop, frame

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
                    return thresh, fgmask_crop, frame_crop, frame, px1, px2, py1, py2


        # snapping with CNT background subtraction implementation
        elif args[0] == 6 and len(args) == 7:
            print("Using CNT Background Subtraction Method")
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
            
            fgbg = cv2.bgsegm.createBackgroundSubtractorCNT()
            frame_skip = 0
            vid.set(cv2.CAP_PROP_POS_FRAMES, frame_skip - 1)

            while True:
                ret, frame = vid.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    fgmask = fgbg.apply(gray)
                else:
                    break
                count = 0
                if frame_skip != 0:
                    for count in range(1,frame_skip):
                        if count == frame_skip:
                            break
                        else:
                            ret, frame = vid.read()

            while True:
                vid.set(cv2.CAP_PROP_POS_FRAMES, frame_no-1)
                ret, frame = vid.read()
                fgmask = fgbg.apply(frame)
                background = fgbg.getBackgroundImage()
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
                    return background, thresh, fgmask_crop, frame_crop, frame

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
                    return thresh, fgmask_crop, frame_crop, frame, px1, px2, py1, py2


        # snapping with CNT background subtraction w/ Shadow Removal
        elif args[0] == 7 and len(args) == 8:
            print("Using CNT Background Subtraction Method w/ Shadow Removal")
            if(isinstance(args[1], cv2.VideoCapture)):
                vid = args[1]
            else:
                print("First argument should be a video")
                return ValueError
            if(isinstance(args[2], str)):
                vid_name = args[2]
            else:
                print("Second argument should be a video name")
                return ValueError
            if(isinstance(args[3], int)):
                frame_no = args[3]
            else:
                print("Third argument should be a frame number")
                return ValueError
            if isinstance(args[4], (float, int)) and isinstance(args[5], (float, int)) or isinstance(args[6], (float, int)) or isinstance(args[7], (float, int)):
                x1 = args[4]
                y1 = args[5]
                x2 = args[6]
                y2 = args[7]
            else:
                print("Argument 4 to 7 should be int or float")
                return ValueError
            
            fgbg = cv2.createBackgroundSubtractorKNN()
            fgbg.setShadowValue(0)

            return_type = "image"
            bgimage = self.get_bgimage(vid, vid_name, frame_skip = 0, return_type = return_type)
            vid.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)
            ret, frame = vid.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if return_type == "array":
                for i in range (0, len(bgimage)):
                    fgmask = fgbg.apply(bgimage[i])
                    
            elif return_type == "image":
                for i in range (0, 30):
                    fgmask = fgbg.apply(bgimage)

            fgmask = fgbg.apply(frame)
            frame_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            fgmask_crop = fgmask[int(y1):int(y2), int(x1):int(x2)]
            thresh = cv2.threshold(fgmask_crop, 45, 0xFF,
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
                return thresh, fgmask_crop, frame_crop, frame

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
                return thresh, fgmask_crop, frame_crop, frame, px1, px2, py1, py2
        

        # snapping combined Grabcut and CNT combined
        elif args[0] == 9 and len(args) == 8:
            print("Using Grabcut with CNT")
            if(isinstance(args[1], cv2.VideoCapture)):
                vid = args[1]
            else:
                print("First argument should be a video")
                return ValueError
            if(isinstance(args[2], str)):
                vid_name = args[2]
            else:
                print("Second argument should be a video name")
                return ValueError
            if(isinstance(args[3], int)):
                frame_no = args[3]
            else:
                print("Third argument should be a frame number")
                return ValueError
            if isinstance(args[4], (float, int)) and isinstance(args[5], (float, int)) or isinstance(args[6], (float, int)) or isinstance(args[7], (float, int)):
                x1 = args[4]
                y1 = args[5]
                x2 = args[6]
                y2 = args[7]
            else:
                print("Argument 4 to 7 should be int or float")
                return ValueError
            
            if self.is_same_video(vid_name) == False: 
                fgbg = bgsegm.createBackgroundSubtractorCNT()

                while True:
                    ret, frame = vid.read()
                    if ret:
                        fgmask = fgbg.apply(frame)
                    else:
                        break
                self.set_previous_video(vid_name, fgbg)

            else:
                print("Using previous fgbg")
                fgbg = self.get_previous_bgimage()
                fgbg.setShadowValue(0)
                
            while True:
                vid.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)
                ret, frame = vid.read()
                fgmask = fgbg.apply(frame)
                frame_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                fgmask_crop = fgmask[int(y1):int(y2), int(x1):int(x2)]
                thresh = cv2.threshold(fgmask_crop, 25, 0xFF,
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
                    return thresh, fgmask_crop, frame_crop, frame

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
                    return thresh, fgmask_crop, frame_crop, frame, px1, px2, py1, py2

        else:
            print("Error in the arguments")
            return ValueError



    def click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("x-coor: ", x, ", y-coor:", y)
