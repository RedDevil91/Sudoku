import cv2
import numpy as np


class ImageRoi(object):
    def __init__(self, contour, moment, approx):
        self.contour = contour
        self.moment = moment
        self.approx = np.reshape(approx, (4, 2)).astype(np.float32)

        sorted_corners = self.approx[np.lexsort((self.approx[:, 0], self.approx[:, 1]))]
        self.old_cord = sorted_corners.copy()
        if sorted_corners[0, 0] > sorted_corners[1, 0]:
            self.old_cord[0], self.old_cord[1] = sorted_corners[1], sorted_corners[0]
        if sorted_corners[2, 0] > sorted_corners[3, 0]:
            self.old_cord[2], self.old_cord[3] = sorted_corners[3], sorted_corners[2]

        # TODO: use the np.sum and np.diff to order the points! diff -> x-y
        diff = np.diff(self.approx, axis=1)
        return


class ImageProcessor(object):
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=np.uint8)
    roi_size = 180
    new_cord = np.float32([[0,          0],
                           [roi_size,   0],
                           [0,          roi_size],
                           [roi_size,   roi_size]])

    def __init__(self):
        self.raw_image = None
        self.preprocessed_image = None
        self.table_image = None
        self.numbers = []
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

        # TODO: use this sorting instead of the other!
        cnts = sorted(contours, key=cv2.contourArea, reverse=True)

        roi_img = None
        for cont in contours:
            perim = cv2.arcLength(cont, True)
            approx = cv2.approxPolyDP(cont, 0.04 * perim, True)

            contour_area = cv2.contourArea(cont)
            hull_area = cv2.contourArea(cv2.convexHull(cont))
            area_ratio = contour_area / float(hull_area)
            if len(approx) == 4 and area_ratio > 0.9:
                M = cv2.moments(cont)
                if roi_img is None or M > roi_img.moment:
                    roi_img = ImageRoi(cont, M, approx)

        cv2.drawContours(blank_image, [roi_img.contour], -1, (255, 255, 255), -1)
        gray = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
        masked_img = cv2.bitwise_and(self.raw_image, self.raw_image, mask=gray)
        masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)

        pers_matrix = cv2.getPerspectiveTransform(roi_img.old_cord, self.new_cord)
        self.table_image = cv2.warpPerspective(masked_img, pers_matrix, (self.roi_size, self.roi_size))
        self.getNumbers()
        return self.table_image

    def drawGrid(self, img):
        cv2.line(img, (self.roi_size/3, 0),   (self.roi_size/3, self.roi_size),     (255, 0, 0), 1)
        cv2.line(img, (2*self.roi_size/3, 0), (2*self.roi_size/3, self.roi_size),   (255, 0, 0), 1)
        cv2.line(img, (0, self.roi_size/3),   (self.roi_size, self.roi_size/3),     (255, 0, 0), 1)
        cv2.line(img, (0, 2*self.roi_size/3), (self.roi_size, 2*self.roi_size/3),   (255, 0, 0), 1)
        return

    def getNumbers(self):
        for row in range(9):
            for col in range(9):
                number = self.table_image[row * self.roi_size / 9:(row + 1) * self.roi_size / 9,
                                          col * self.roi_size / 9:(col + 1) * self.roi_size / 9]
                _, number = cv2.threshold(number, 110, 255, cv2.THRESH_BINARY_INV)
                self.numbers.append(number)
        return

if __name__ == '__main__':
    import time
    import sys
    import matplotlib.pyplot as plt

    # ****************************************************************
    input_img = cv2.imread('test_img2.jpg')

    start = time.time()
    processor = ImageProcessor()
    processor.new_image(input_img)
    processor.preprocess()
    out = processor.search_table()
    end = time.time()
    print "Process time: %f" % (end-start)

    plt.imshow(out)
    plt.show()
    for idx, number in enumerate(processor.numbers):
        cv2.imwrite('numbers/numbers%d.jpg' % idx, number)
    # ****************************************************************

    # cap = cv2.VideoCapture(0)
    #
    # if not cap.isOpened():
    #     print "Failed to open caption!"
    #     sys.exit(1)
    #
    # processor = ImageProcessor()
    #
    # while (True):
    #     # Capture frame-by-frame
    #     _, frame = cap.read()
    #
    #     processor.new_image(frame)
    #     processor.preprocess()
    #     out = processor.search_table()
    #
    #     # Display the resulting frame
    #     cv2.imshow('frame', out)
    #
    #     key = cv2.waitKey(1) & 0xFF
    #     if key == ord('q'):
    #         break
    #     elif key == ord('c'):
    #         import glob
    #         cv2.imwrite('test_img%d.jpg' % len(glob.glob('*.jpg')), frame)
    #
    # # When everything done, release the capture
    # cap.release()
    # cv2.destroyAllWindows()
