import numpy as np
import cv2

img = cv2.imread('4.jpg')
scale = 2 #downscale by 
dim = (int(1920/scale),int(1080/scale))
img = cv2.resize(img,dim)
# img = cv2.bilateralFilter(img,9,75,75)
# img = cv2.blur(img,(5,5))
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

while True:
    r = cv2.selectROI('Select Area',img)
    cv2.destroyWindow('Select Area')
    rect = r
    print(rect)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    cv2.imshow("Image",img)
    # cv2.imshow("Mask",mask2)
    key = cv2.waitKey(0)
    if key == ord('q'):
        cv2.destroyAllWindows()
        break
    else:
        print('restarting')