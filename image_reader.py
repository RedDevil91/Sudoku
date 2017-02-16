import cv2
import sys
import numpy as np

kernel = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]], dtype=np.uint8)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print "Failed to open caption!"
    sys.exit(1)

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
    thresholded = cv2.bitwise_not(thresholded)
    dilated = cv2.dilate(thresholded, kernel)

    # Display the resulting frame
    cv2.imshow('frame', dilated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
