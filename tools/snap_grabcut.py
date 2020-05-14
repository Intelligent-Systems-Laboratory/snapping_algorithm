import cv2
import numpy as np
import imutils


def create_mask(v_ex):
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

    return mask, rect


def grabcut(img, mask, rect):
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,25,cv2.GC_BGD) 
    display = img.copy()
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    grabcut_crop = img*mask2[:,:,np.newaxis]
    img2 = np.where(grabcut_crop!=0,255, grabcut_crop).astype('uint8')
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(img2,127,255,cv2.THRESH_BINARY)[1]
    return thresh