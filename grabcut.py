import numpy as np
import cv2

img = cv2.imread('3.jpg')
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
    # print(rect)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,10,cv2.GC_INIT_WITH_RECT)
    orig = img.copy()   
    display = orig.copy()
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    img2  = np.where(img!=0,255,img).astype('uint8')
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) 
    img3 = cv2.threshold(img2,127,255,cv2.THRESH_BINARY)
    print(np.shape(img3))
    # img2 = np.where(img!=0,img*[255,255,255],[0,0,0])
    # img2 = np.where()
    # print(img)

    # cv2.imshow("Image",img)
    cv2.imshow("Image1",img)
    cv2.imshow("Image2",img2)
    # cv2.imshow("Image3",img3)

    # cv2.imshow("Mask",mask2)

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
    cv2.imshow('crop', display)

    key = cv2.waitKey(0)
    if key == ord('q'):
        cv2.destroyAllWindows()
        break
    else:
        img = orig
        print('restarting')
        cv2.destroyAllWindows() 