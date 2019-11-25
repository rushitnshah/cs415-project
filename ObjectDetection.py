import cv2
import numpy as np

vid = cv2.VideoCapture(0)

lower_green = np.array([33,80,40])
upper_green = np.array([102,255,255])

kernelOpen = np.ones((3,3))
kernelClose = np.ones((5,5))

# font = cv2.cv.InitFont(cv2.FONT_HERSHEY_SIMPLEX,2,0.5,0,3,1)
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (0,255,255)

while True:

    ret, frame = vid.read()
    frame = cv2.resize(frame,(340,220))

    #Convert frame to HSV
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Create mask
    frameMaskG = cv2.inRange(frameHSV, lower_green, upper_green)

    #Morphology
    frameMaskOpen = cv2.morphologyEx(frameMaskG,cv2.MORPH_OPEN,kernelOpen)
    frameMaskClose = cv2.morphologyEx(frameMaskOpen,cv2.MORPH_CLOSE,kernelClose)

    #Final Mask
    maskFinal = frameMaskClose

    #Find Object
    contours, hierarchy = cv2.findContours(maskFinal.copy(),
                                            cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(frame,contours,-1,(255,0,0),1)
    for i in range(len(contours)):
        x,y,w,h = cv2.boundingRect(contours[i])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(frame,str(i+1),(x,y+h),fontFace,
                                            fontScale,
                                            fontColor)

    #Plot
    cv2.imshow("Original",frame)
    cv2.imshow("Open Mask",frameMaskClose)
    cv2.imshow("Green Mask",frameMaskG)

    cv2.waitKey(5)
