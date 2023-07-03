from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing import image
import cvlib as cv
                    
# load model
model = load_model('gender_detection.model')

# open webcam
webcam = cv2.VideoCapture(0)
    
classes = ['man','woman']

# loop through frames
while webcam.isOpened():

    # read frame from webcam 
    status, frame = webcam.read()
    frame = cv2.resize(frame,(96,96))
    frame = frame.astype("float") / 255.0
    x = img_to_array(frame)
    x = np.expand_dims(frame, axis=0)

        # apply gender detection on face
    val = model.predict(x)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

	
    print(val)

    # display output
    cv2.imshow("gender detection", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()