import cv2
import sys

imagePath = sys.argv[1]

image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=3,
    minSize=(30, 30)
)

# print(faces)
x, y, w, h = faces[0]

    
roi_color = gray[y:y + h, x:x + w]
roi_color = cv2.resize(roi_color, (48, 48))

cv2.imwrite(str(w) + str(h) + '_faces.jpg', roi_color)

