import cv2
import numpy as np
from pynput.keyboard import Key, Controller
import wx


keyboard = Controller()
app   = wx.App(False)
(sx,sy) = wx.GetDisplaySize()
(resx,resy) = (340,220)

lower_green = np.array([33,80,40])
upper_green = np.array([102,255,255])

lower_red1 = np.array([0,120,80]) #example value
upper_red1 = np.array([6,255,255]) #example value
lower_red2 = np.array([175,120,80])
upper_red2 = np.array([180,255,255])

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
    frameMaskOpenG = cv2.morphologyEx(frameMaskG,cv2.MORPH_OPEN,kernelOpen)
    frameMaskCloseG = cv2.morphologyEx(frameMaskOpenG,cv2.MORPH_CLOSE,kernelClose)
    #Final Mask
    maskFinalG = frameMaskCloseG
    #Find Object
    _, contoursG, hierarchyG = cv2.findContours(maskFinalG.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    #Create mask
    frameMaskR1 = cv2.inRange(frameHSV, lower_red1, upper_red1)
    frameMaskR2 = cv2.inRange(frameHSV, lower_red2, upper_red2)
    frameMaskR  = frameMaskR1 #+ frameMaskR2
    #Morphology
    frameMaskOpenR = cv2.morphologyEx(frameMaskR,cv2.MORPH_OPEN,kernelOpen)
    frameMaskCloseR = cv2.morphologyEx(frameMaskOpenR,cv2.MORPH_CLOSE,kernelClose)
    #Final Mask
    maskFinalR = frameMaskCloseR
    #Find Object
    _, contoursR, hierarchyR = cv2.findContours(maskFinalR.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)


    # cv2.drawContours(frame,contoursG,-1,(0,255,0),1)
    # for i in range(len(contoursG)):
    #     x,y,w,h = cv2.boundingRect(contoursG[i])
    #     cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    #     cv2.putText(frame,"Green "+str(i+1),(x,y+h),fontFace,
    #                                         fontScale,
    #                                         fontColor)
    #
    # cv2.drawContours(frame,contoursR,-1,(0,0,255),1)
    # for i in range(len(contoursR)):
    #     x,y,w,h = cv2.boundingRect(contoursR[i])
    #     cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    #     cv2.putText(frame,"Red "+str(i+1),(x,y+h),fontFace,
    #                                         fontScale,
    #                                         fontColor)
    if(len(contoursG)>0):
        # if (pinchFlag is 1):
        #     pinchFlag=0
        #     mouse.release(Button.left)
        x1,y1,w1,h1=cv2.boundingRect(contoursG[0])
        cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(255,0,0),2)
        # cx1 = round(x1+w1/2)
        # cy1 = round(y1+h1/2)
        # cx2 = round(x2+w2/2)
        # cy2 = round(y2+h2/2)
        # cx  = round((cx1+cx2)/2)
        # cy  = round((cy1+cy2)/2)
        # cv2.line(frame,(cx1,cy1),(cx2,cy2),(255,0,0),1)
        # cv2.circle(frame,(cx,cy),2,(0,0,255),1)
        # mLoc = mLocOld + ((cx,cy)-mLocOld)/damp
        # mouse.position=(sx-(mLoc[0]*sx/resx),mLoc[1]*sy/resy)
        # mLocOld=mLoc
        # openx,openy,openw,openh = cv2.boundingRect(np.array([[[x1,y1],
        #                                                     [x1+w1,y1+h1],
        #                                                     [x2,y2],
        #                                                     [x2+w2,y2+h2]]]))
        # cv2.rectangle(frame,(openx,openy),(openx+openw,openy+openw),(255,0,0),2)
        keyboard.press(Key.up)
        keyboard.release(Key.up)

    if(len(contoursR)>0):
        # if (pinchFlag is 1):
        #     pinchFlag=0
        #     mouse.release(Button.left)
        x1,y1,w1,h1=cv2.boundingRect(contoursR[0])
        cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(255,0,0),2)
        # cx1 = round(x1+w1/2)
        # cy1 = round(y1+h1/2)
        # cx2 = round(x2+w2/2)
        # cy2 = round(y2+h2/2)
        # cx  = round((cx1+cx2)/2)
        # cy  = round((cy1+cy2)/2)
        # cv2.line(frame,(cx1,cy1),(cx2,cy2),(255,0,0),1)
        # cv2.circle(frame,(cx,cy),2,(0,0,255),1)
        # mLoc = mLocOld + ((cx,cy)-mLocOld)/damp
        # mouse.position=(sx-(mLoc[0]*sx/resx),mLoc[1]*sy/resy)
        # mLocOld=mLoc
        # openx,openy,openw,openh = cv2.boundingRect(np.array([[[x1,y1],
        #                                                     [x1+w1,y1+h1],
        #                                                     [x2,y2],
        #                                                     [x2+w2,y2+h2]]]))
        # cv2.rectangle(frame,(openx,openy),(openx+openw,openy+openw),(255,0,0),2)
        keyboard.press(Key.down)
        keyboard.release(Key.down)

    # elif(len(contours)==1):
    #     x,y,w,h=cv2.boundingRect(contours[0])
    #     if (pinchFlag is 0):
    #         if abs((w*h - openw*openh)/(h*w))*100 < 10:
    #             pinchFlag=1
    #             mouse.press(Button.left)
    #             openx,openy,openw,openh = (0,0,0,0)
    #     else:
    #         cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    #         cx = round(x+w/2)
    #         cy = round(y+h/2)
    #         cv2.circle(frame,(cx,cy),round((w+h)/4),(0,0,255),1)
    #         mLoc = mLocOld + ((cx,cy)-mLocOld)/damp
    #         mouse.position=(sx-(mLoc[0]*sx/resx),mLoc[1]*sy/resy)
    #         mLocOld=mLoc


    #Plot
    cv2.imshow("Original",frame)
    cv2.waitKey(5)
