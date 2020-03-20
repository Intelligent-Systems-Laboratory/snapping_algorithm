import cv2
import numpy as np
import imutils
import argparse
import os.path
from os import path

# import this class to use the functions

class Snap:
    def __init__ (self):
        self.help_message = \
            '''
        Usage of this algorithm:
        snap(img, x1, y1, x2, y2) -- purely vision-based
        snap(prev, img, x1, y1, x2, y2) -- optical flow implementation

        '''
        self.SNAP_THRESHOLD = 1
        self.SNAP_OPTICAL_FLOW = 2
        self.SNAP_BACKGROUND_SUBTRACTION = 3
    

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
        elif args[0] == 3 and len(args) == 8:
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
            if(isinstance(args[7], str)):
                folder_name = args[7]
            else:
                print("Folder name should be a string")
                return ValueError
            
            fgbg = cv2.createBackgroundSubtractorKNN()
            fgbg.setShadowValue(0)
            
            if frame_no > 500:
                vid.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 500)
            else:
                vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            while True:
                ret, frame = vid.read()
                fgmask = fgbg.apply(frame)
                if int(vid.get(cv2.CAP_PROP_POS_FRAMES)) > frame_no:
                    frame_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    cv2.imwrite(folder_name+'/frame_'+str(frame_no)+'_frame_crop.jpg', frame_crop)
                    fgmask_crop = fgmask[int(y1):int(y2), int(x1):int(x2)]
                    cv2.imwrite(folder_name+'/frame_'+str(frame_no)+'_fgmask_crop.jpg', fgmask_crop)
                    thresh = cv2.threshold(fgmask_crop, 20, 0xFF,
                                            cv2.THRESH_BINARY)[1]
                    cv2.imwrite(folder_name+'/frame_'+str(frame_no)+'_thresh.jpg', thresh)
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
                        cv2.rectangle(frame_crop, (nX, nY), (nX + w, nY + h), (0, 0xFF, 0), 4)
                        cv2.imwrite(folder_name+'/frame_'+str(frame_no)+'_crop.jpg', frame_crop)
                        return x1 + nX, y1 + nY, x1 + nX + w, y1 + nY + h

        
        else:
            print(self.help_message)
            return ValueError



    def click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("x-coor: ", x, ", y-coor:", y)

    def run(self,vid_name,FRAME_NO,bboxx1,bboxy1,bboxx2,bboxy2):
        # the video could only use up to 400 frames for high accuracy
        vid = cv2.VideoCapture(vid_name)
        
        # THE PARAMETERS TO CHANGE
        # Coordinates of the resized image, not the full size
        vid.set(1, FRAME_NO)
        ret, img = vid.read()
        (H,W) = img.shape[:2]
        
        imgvis = imutils.resize(img, width=1028)
        folder_name = vid_name[7:-4]
        ratioW = W/1028

        x1, y1, x2, y2 = snap.snap_algorithm(snap.SNAP_BACKGROUND_SUBTRACTION, vid, FRAME_NO, bboxx1*ratioW, bboxy1*ratioW, bboxx2*ratioW, bboxy2*ratioW, folder_name)
        x1 = int(x1 / ratioW)
        x2 = int(x2 / ratioW)
        y1 = int(y1 / ratioW)
        y2 = int(y2 / ratioW)

        cv2.rectangle(imgvis, (x1, y1), (x2, y2), (0, 0xFF, 0), 4)
        cv2.imwrite(folder_name+'/frame_'+str(FRAME_NO)+'.jpg',imgvis)
            

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--run_from_file", required=False, default="areas_used.txt", help="process coordinates from file")
    args = vars(ap.parse_args())

    snap = Snap()
    i = 1
    with open (args["run_from_file"], 'rt') as myfile:
        for line in myfile:
            if i%8 == 1:
                video = line.rstrip('\n').replace('vid name: ','')
                folder_name = video[7:-4]
                print('folder name: ' + folder_name)
                if path.exists(folder_name) == False:
                    os.mkdir(folder_name)
                    print("directory made")
            if i%8 == 2:
                frame_no = int(line.rstrip('\n').replace('frame no: ',''))
            if i%8 == 3:
                x1 = int(line.rstrip('\n').replace('x1: ',''))
            if i%8 == 4:
                y1 = int(line.rstrip('\n').replace('y1: ',''))
            if i%8 == 5:
                x2 = int(line.rstrip('\n').replace('x2: ',''))
            if i%8 == 6:
                y2 = int(line.rstrip('\n').replace('y2: ',''))
            if i%8 == 7:
                print(video)
                print(frame_no)
                print(int(x1))
                print(int(y1))
                print(int(x2))
                print(int(y2))
            if i%8 == 0:
                snap.run(video,frame_no,x1,y1,x2,y2)
                print("\n\n")
            i = i+1
                
 
