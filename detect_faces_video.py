# python detect_faces_video.py 

print("Step 1: Importing libraries")

from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2

print("Step 2: Creating function to load the arguments")

def load_args():
    args = {
                'prototxt': 'files/deploy.prototxt.txt', 
                'model': 'model/res10_300x300_ssd_iter_140000.caffemodel', 
                'confidence': 0.5
    }
    
    return args
    
args = load_args()

# print(args)
print("Step 3: Loading the model")

net = cv2.dnn.readNetFromCaffe(
    args["prototxt"], 
    args["model"]
)

print("Step 4: Creating object to VideoStream class")

vs = VideoStream(src = 0).start()
time.sleep(2.0)

print("Step 5: Video means, video stream - frames")

while True:

    print("Step 5: Read the video frame and resize it")

    frame = vs.read()
    frame = imutils.resize(frame, width = 400)
 
    print("Step 6: Access the fram dimensions and convert into blob")

    (h, w) = frame.shape[:2]
        
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 
        1.0,
        (300, 300), 
        (104.0, 177.0, 123.0)
    )
    print("Step 6: set blob object to the model to do the detection")

    net.setInput(blob)
    detections = net.forward()

    print("Step 7: set blob object to the model to do the detection")

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence < args["confidence"]:
            continue

        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # draw the bounding box of the face along with the associated
        # probability
        
        text = "{:.2f}%".format(confidence * 100)
        
        y = startY - 10 if startY - 10 > 10 else startY + 10
        
        cv2.rectangle(  
                        frame, 
                        (startX, startY), 
                        (endX, endY),
                        (0, 0, 255), 2
        )
        
        cv2.putText(
                        frame, 
                        text, 
                        (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.45, 
                        (0, 0, 255), 
                        2
        )

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

