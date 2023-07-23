import cv2
import numpy as np
haar_cascade_face = cv2.CascadeClassifier('Haarcascade_frontalface_default.xml')
haar_cascade_eye = cv2.CascadeClassifier('haarcascade_eye.xml')
haar_cascade_smile = cv2.CascadeClassifier('haarcascade_smile.xml')
# define a video capture object
vid = cv2.VideoCapture(0)

while True:

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # Converting image to grayscale
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Applying the face detection method on the grayscale image
    faces_rect = haar_cascade_face.detectMultiScale(gray_img, 1.1, 9)

    # Iterating through rectangles of detected faces
    for (x, y, w, h) in faces_rect:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        eyes_rect = haar_cascade_eye.detectMultiScale(gray_img, 1.1, 4)
        for (x1, y1, w1, h1) in eyes_rect:
            # radius = np.sqrt((w1//2)**2 + (h1//2)**2).astype('int32')
            # cv2.circle(frame, (w1//2 + x1, h1//2 + y1), radius, (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)

    cv2.imshow('Detected faces', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
