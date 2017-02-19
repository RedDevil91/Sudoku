import cv2
import sys
import numpy as np


class ImageProcessor(object):
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=np.uint8)

    def __init__(self):
        self.raw_image = None
        self.preprocessed_image = None
        self.out_image = None
        return

    def new_image(self, image):
        self.raw_image = image
        return

    def preprocess(self):
        gray = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
        thresholded = cv2.bitwise_not(thresholded)
        self.preprocessed_image = cv2.dilate(thresholded, self.kernel)
        return

    def search_table(self):
        img, contours, hierarchy = cv2.findContours(self.preprocessed_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blank_image = np.zeros((480, 640, 3), dtype=np.uint8)

        max_c = None
        max_mom = 0
        for cont in contours:
            perim = cv2.arcLength(cont, True)
            approx = cv2.approxPolyDP(cont, 0.04 * perim, True)
            if len(approx) == 4:
                M = cv2.moments(cont)
                if max_c is None or M > max_mom:
                    max_c = cont
                    max_mom = M
        cv2.drawContours(blank_image, [max_c], -1, (255, 255, 255), -1)
        gray = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
        out = cv2.bitwise_and(self.raw_image, self.raw_image, mask=gray)
        return out

if __name__ == '__main__':

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print "Failed to open caption!"
        sys.exit(1)

    processor = ImageProcessor()

    while (True):
        # Capture frame-by-frame
        _, frame = cap.read()

        processor.new_image(frame)
        processor.preprocess()
        out = processor.search_table()

        # Display the resulting frame
        cv2.imshow('frame', out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
