import cv2
import numpy as np
import tensorflow as tf
from pynput.keyboard import Key, Controller
import wx

keyboard = Controller()
app   = wx.App(False)
(sx,sy) = wx.GetDisplaySize()
(resx,resy) = (340,220)

MODEL_NAME = 'handtracking-master/hand_inference_graph'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

#Threshold for an object to be classified as hand
score_thresh = 0.2

#Maximum number of hands tot detect
NUM_HANDS = 2

fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (77, 255, 9)

im_width = 340
im_height = 220
# Load a frozen infrerence graph into memory
def load_inference_graph():

    # load frozen tensorflow model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print(">>>Tensorflow graph loaded.<<<")
    return detection_graph, sess

if __name__ == '__main__':
    # Load TF inference graph
    detection_graph, sess = load_inference_graph()
    sess = tf.Session(graph=detection_graph)

    #Setup video capture pbject
    vid = cv2.VideoCapture(0)
    ret,frameBGR = vid.read()
    frameBGR = cv2.resize(frameBGR,(im_width,im_height))
    frameBGR = cv2.flip(frameBGR, 1)
    frame = cv2.cvtColor(frameBGR, cv2.COLOR_BGR2RGB)

    while True:
        if (frame is not None):
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name(
                'detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name(
                'detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name(
                'detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')

            image_np_expanded = np.expand_dims(frame, axis=0)

            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores,
                    detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            boxes  = np.squeeze(boxes)
            scores = np.squeeze(scores)

            hands_detected = 0
            P1_list = []
            P2_list = []
            C_list  = []
            for i in range(NUM_HANDS):
                if (scores[i] > score_thresh):
                    hands_detected += 1
                    (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                                  boxes[i][0] * im_height, boxes[i][2] * im_height)
                    p1 = (int(left), int(top))
                    p2 = (int(right), int(bottom))
                    P1_list.append(p1)
                    P2_list.append(p2)

                    x = int((left+right)/2)
                    y = int((top+bottom)/2)
                    C_list.append((x,y))

                    cv2.circle(frame,(x,y),6,(77, 255, 9), 2)
                    cv2.rectangle(frame, p1, p2, (77, 255, 9), 2, 1)
                    cv2.putText(frame,str(hands_detected),(x,y),fontFace,fontScale,fontColor)

            #If two hands are dettected
            if (len(P1_list)==2):
                keyboard.press(Key.space)
                keyboard.release(Key.space)
                cv2.putText(frame,"Fire!",(round(im_width/2),round(0.10*im_height)),fontFace,fontScale,fontColor)


            elif(len(P1_list)==1):
                x1 = P1_list[0][0]
                y1 = P1_list[0][1]
                x2 = P2_list[0][0]
                y2 = P2_list[0][1]

                cx = round((x1+x2)/2)
                cy = round((y1+y2)/2)

                if cx < 0.3*im_width:
                    for _ in range(80):
                        keyboard.press(Key.left)
                        keyboard.release(Key.left)
                        cv2.putText(frame,"Left",(round(im_width/2),round(0.10*im_height)),fontFace,fontScale,fontColor)

                elif cx > 0.7*im_width:
                    for _ in range(80):
                        keyboard.press(Key.right)
                        keyboard.release(Key.right)
                        cv2.putText(frame,"Right",(round(im_width/2),round(0.10*im_height)),fontFace,fontScale,fontColor)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Simple Frame",frame)

        #Read next frame
        ret,frameBGR = vid.read()
        frameBGR = cv2.resize(frameBGR,(340,220))

        #Mirror image for sake of sanity
        frameBGR = cv2.flip(frameBGR, 1)

        #Convert to RGB
        frame = cv2.cvtColor(frameBGR, cv2.COLOR_BGR2RGB)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    sess.close()
    cv2.destroyAllWindows()
