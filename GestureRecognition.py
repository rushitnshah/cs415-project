import cv2
import numpy as np
from pynput.mouse import Button, Controller
import wx


mouse = Controller()
app   = wx.App(False)
(sx,sy) = wx.GetDisplaySize()
(resx,resy) = (340,220)

lower_green = np.array([33,80,40])
upper_green = np.array([102,255,255])

kernelOpen = np.ones((3,3))
kernelClose = np.ones((5,5))

# font = cv2.cv.InitFont(cv2.FONT_HERSHEY_SIMPLEX,2,0.5,0,3,1)
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (0,255,255)

vid  = cv2.VideoCapture(0)
vid.set(3,resx)
vid.set(4,resy)

mLocOld = np.array([0,0])
mLoc    = np.array([0,0])
damp    = 2
pinchFlag = 0
openx,openy,openw,openh = (0,0,0,0)
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
    _, contours, hierarchy = cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(frame,contours,-1,(255,0,0),1)

    if(len(contours)==2):
        if (pinchFlag is 1):
            pinchFlag=0
            mouse.release(Button.left)
        x1,y1,w1,h1=cv2.boundingRect(contours[0])
        x2,y2,w2,h2=cv2.boundingRect(contours[1])
        cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(255,0,0),2)
        cv2.rectangle(frame,(x2,y2),(x2+w2,y2+h2),(255,0,0),2)
        cx1 = round(x1+w1/2)
        cy1 = round(y1+h1/2)
        cx2 = round(x2+w2/2)
        cy2 = round(y2+h2/2)
        cx  = round((cx1+cx2)/2)
        cy  = round((cy1+cy2)/2)
        cv2.line(frame,(cx1,cy1),(cx2,cy2),(255,0,0),1)
        cv2.circle(frame,(cx,cy),2,(0,0,255),1)
        mLoc = mLocOld + ((cx,cy)-mLocOld)/damp
        mouse.position=(sx-(mLoc[0]*sx/resx),mLoc[1]*sy/resy)
        mLocOld=mLoc
        openx,openy,openw,openh = cv2.boundingRect(np.array([[[x1,y1],
                                                            [x1+w1,y1+h1],
                                                            [x2,y2],
                                                            [x2+w2,y2+h2]]]))
        # cv2.rectangle(frame,(openx,openy),(openx+openw,openy+openw),(255,0,0),2)
    elif(len(contours)==1):
        x,y,w,h=cv2.boundingRect(contours[0])
        if (pinchFlag is 0):
            if abs((w*h - openw*openh)/(h*w))*100 < 10:
                pinchFlag=1
                mouse.press(Button.left)
                openx,openy,openw,openh = (0,0,0,0)
        else:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            cx = round(x+w/2)
            cy = round(y+h/2)
            cv2.circle(frame,(cx,cy),round((w+h)/4),(0,0,255),1)
            mLoc = mLocOld + ((cx,cy)-mLocOld)/damp
            mouse.position=(sx-(mLoc[0]*sx/resx),mLoc[1]*sy/resy)
            mLocOld=mLoc


    #Plot
    cv2.imshow("Original",frame)

    cv2.waitKey(5)
