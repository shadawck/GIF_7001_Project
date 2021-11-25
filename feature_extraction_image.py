import cv2
import numpy as np
import dlib

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

img = cv2.imread("face.jpg")

gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

faces = detector(gray)
for face in faces:
    x1 = face.left() 
    y1 = face.top() 
    x2 = face.right() 
    y2 = face.bottom() 

    landmarks = predictor(image=gray, box=face)

    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y

        cv2.circle(img=img, center=(x, y), radius=1, color=(0, 255, 0), thickness=-1)

cv2.imshow(winname="Face", mat=img)

cv2.waitKey(delay=0)

cv2.destroyAllWindows()