import cv2
import numpy as np


class RegionOfInterest(object):
    def __init__(self, contour, moment, approx):
        self.contour = contour
        self.moment = moment
        self.approx = np.reshape(approx, (4, 2)).astype(np.float32)

        # sorting corners for perspective transform
        self.corners = np.zeros(self.approx.shape, dtype=np.float32)
        corner_diff = np.diff(self.approx, axis=1)
        corner_sum = np.sum(self.approx, axis=1)
        self.corners[0] = self.approx[np.argmin(corner_sum)]
        self.corners[1] = self.approx[np.argmin(corner_diff)]
        self.corners[2] = self.approx[np.argmax(corner_sum)]
        self.corners[3] = self.approx[np.argmax(corner_diff)]
        return


class ImageProcessor(object):
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=np.uint8)
    horizontal_kernel = np.ones((1, 50), dtype=np.uint8)
    vertical_kernel = np.ones((50, 1), dtype=np.uint8)

    roi_size = 28 * 9
    corners = np.float32([[0,          0],
                          [roi_size,   0],
                          [roi_size,   roi_size],
                          [0,          roi_size]])

    def __init__(self):
        self.raw_image = None
        self.preprocessed_image = None
        self.table_image = None
        self.numbers = []
        return

    def new_image(self, image):
        self.raw_image = image
        self.preprocessed_image = self.preProcessImage(self.raw_image)
        self.search_table()
        return

    def preProcessImage(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 2)
        return cv2.dilate(thresholded, self.kernel)

    def search_table(self):
        roi = self.getSquares(self.preprocessed_image)

        pers_matrix = cv2.getPerspectiveTransform(roi.corners, self.corners)
        self.table_image = cv2.warpPerspective(self.raw_image, pers_matrix, (self.roi_size, self.roi_size))

        # cv2.imshow('orig', self.table_image)

        image = cv2.cvtColor(self.table_image, cv2.COLOR_BGR2GRAY)
        image = cv2.adaptiveThreshold(image, 255,
                                      cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, -2)

        vertical = cv2.erode(image, self.horizontal_kernel)
        vertical = cv2.dilate(vertical, self.horizontal_kernel)
        # vertical = cv2.dilate(vertical, self.horizontal_kernel)

        cv2.imshow('vertical', vertical)

        horizontal = cv2.erode(image, self.vertical_kernel)
        horizontal = cv2.dilate(horizontal, self.vertical_kernel)
        # horizontal = cv2.dilate(horizontal, self.vertical_kernel)

        cv2.imshow('horizontal', horizontal)

        image = cv2.bitwise_and(vertical, horizontal)

        # cv2.imshow('points', self.table_image)

        img, contours, hierarchy = cv2.findContours(image.copy(),
                                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cont in contours:
            mom = cv2.moments(cont)
            try:
                x, y = int(mom['m10']/mom['m00']), int(mom['m01']/mom['m00'])
                cv2.circle(self.table_image, (x, y), 4, (0, 255, 0), -1)
            except ZeroDivisionError:
                print 'ZeroError!'

        # self.getNumbers()
        # self.drawGrid(self.table_image)
        return self.table_image

    def getSquares(self, image):
        img, contours, hierarchy = cv2.findContours(image.copy(),
                                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # sort the contours
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cont in contours:
            perim = cv2.arcLength(cont, True)
            approx = cv2.approxPolyDP(cont, 0.04 * perim, True)

            contour_area = cv2.contourArea(cont)
            hull_area = cv2.contourArea(cv2.convexHull(cont))
            area_ratio = contour_area / float(hull_area)
            if len(approx) == 4 and area_ratio > 0.9:
                M = cv2.moments(cont)
                roi_img = RegionOfInterest(cont, M, approx)
                break
        else:
            # if there is no square
            roi_img = None
        return roi_img

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
                # number = number[3:24, 3:24]
                # number, contours, hierarchy = cv2.findContours(number, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # contours = sorted(contours, key=cv2.contourArea, reverse=True)
                # number = cv2.drawContours(number, contours, 0, (255, 255, 255), 1)
                self.numbers.append(number)
        return

if __name__ == '__main__':
    import time
    import sys

    def load_from_file(filename):
        input_img = cv2.imread(filename)

        start = time.time()
        processor = ImageProcessor()
        processor.new_image(input_img)
        end = time.time()
        print "Process time: %f" % (end-start)

        cv2.imshow('Table', processor.table_image)
        cv2.waitKey()
        for idx, number in enumerate(processor.numbers):
            cv2.imwrite('numbers/numbers%d.jpg' % idx, number)

    def load_from_camera(camera_number):
        cap = cv2.VideoCapture(camera_number)

        if not cap.isOpened():
            print "Failed to open caption!"
            sys.exit(1)

        processor = ImageProcessor()

        while (True):
            # Capture frame-by-frame
            _, frame = cap.read()

            processor.new_image(frame)

            # Display the resulting frame
            cv2.imshow('frame', processor.table_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                import glob
                cv2.imwrite('test_img%d.jpg' % len(glob.glob('*.jpg')), frame)

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

    load_from_file('test_img0.jpg')
    # load_from_camera(0)
