import cv2
import numpy as np
from keras.models import model_from_json


class RegionOfInterest(object):
    padding = 3

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

        # add padding to table
        self.corners[0][0] -= self.padding
        self.corners[0][1] -= self.padding
        self.corners[1][0] += self.padding
        self.corners[1][1] -= self.padding
        self.corners[2][0] += self.padding
        self.corners[2][1] += self.padding
        self.corners[3][0] -= self.padding
        self.corners[3][1] += self.padding
        return


class Number(object):
    def __init__(self, raw_image, bl_corner, number):
        self.image = raw_image
        x, y = bl_corner
        self.bl_corner = x, y
        self.number = number
        return


class ImageProcessor(object):
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=np.uint8)
    # TODO: create constant from these values
    kernel_line_size = 25
    number_size = 36
    number_padding = 4
    roi_size = number_size * 9
    grid_tolerance = 7

    horizontal_kernel = np.ones((1, kernel_line_size), dtype=np.uint8)
    vertical_kernel = np.ones((kernel_line_size, 1), dtype=np.uint8)

    corners = np.float32([[0,          0],
                          [roi_size,   0],
                          [roi_size,   roi_size],
                          [0,          roi_size]])

    number_corners = np.float32([[0,            0],
                                 [number_size,  0],
                                 [number_size,  number_size],
                                 [0,            number_size]])

    def __init__(self):
        self.raw_image = None
        self.preprocessed_image = None
        self.table_image = None
        self.numbers = []

        # load model
        with open('mnist_model.json', 'r+') as json_file:
            loaded_json_model = json_file.read()
        self.model = model_from_json(loaded_json_model)
        self.model.load_weights('mnist_model_weights.h5')
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return

    def new_image(self, image):
        self.raw_image = image
        self.preprocessed_image = self.preProcessImage(self.raw_image)
        out_img = self.search_table()
        return out_img

    def preProcessImage(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        return cv2.dilate(thresholded, self.kernel)

    def search_table(self):
        # get the sudoku table
        roi = self.getSquares(self.preprocessed_image)
        if not roi:
            return self.raw_image

        # use perspective transformation on the image to get the table only
        pers_matrix = cv2.getPerspectiveTransform(roi.corners, self.corners)
        self.table_image = cv2.warpPerspective(self.raw_image, pers_matrix, (self.roi_size, self.roi_size))

        # find grid points
        grid_points = self.findGridPoints()
        # find the defects in grid points
        filtered_rows = self.filterGridPoints(grid_points)
        filtered_cols = self.filterGridPoints(grid_points, by_rows=False)

        row_points = [point for grid_line in filtered_rows for point in grid_line]
        col_points = [point for grid_line in filtered_cols for point in grid_line]
        points = []
        for point in row_points:
            if point in col_points:
                points.append(point)
        print len(points)

        self.numbers = self.getNumbers(points)

        for i, point in enumerate(points):
            x, y = point
            cv2.circle(self.table_image, (x, y), 2, (0, 255, 0), -1)
            # cv2.putText(self.table_image, str(i), (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))
        for number in self.numbers:
            cv2.putText(self.table_image, str(number.number), number.bl_corner,
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))
        return self.table_image

    def findGridPoints(self):
        # remove the grid points from the previous loop
        grid_points = []
        # use inverse adaptive binary threshold to preprocess the table
        _image = cv2.cvtColor(self.table_image, cv2.COLOR_BGR2GRAY)
        image = cv2.adaptiveThreshold(_image, 255,
                                      cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, -2)

        cv2.imshow('threshold', image)
        # find the vertical and the horizontal lines on the image
        vertical = cv2.morphologyEx(image, cv2.MORPH_OPEN, self.horizontal_kernel)
        horizontal = cv2.morphologyEx(image, cv2.MORPH_OPEN, self.vertical_kernel)
        # the grid points will be the intersections of the horizontal and vertical lines
        image = cv2.bitwise_and(vertical, horizontal)

        # show lines for debug reason
        cv2.imshow('vertical', vertical)
        cv2.imshow('horizontal', horizontal)
        cv2.imshow('grid', image)

        img, contours, hierarchy = cv2.findContours(image.copy(),
                                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cont in contours:
            mom = cv2.moments(cont)
            try:
                x, y = int(mom['m10'] / mom['m00']), int(mom['m01'] / mom['m00'])
                grid_points.append((x, y))
            except ZeroDivisionError:
                # add the centroids when there is zero division error (the contour is a line)
                points = np.ravel(cont)
                grid_points.append((points[0], points[1]))
        return grid_points

    def filterGridPoints(self, grid_points, by_rows=True):
        grid_lines = []
        # sort the grid points by the y coord
        grid_points.sort(key=lambda point: point[1] if by_rows else point[0])
        # group the points according to the x or y coords
        for x, y in grid_points:
            for grid_line in grid_lines:
                if not grid_line:
                    continue
                point = grid_line[0]
                difference = abs(y - point[1]) if by_rows else abs(x - point[0])
                if difference <= self.grid_tolerance:
                    grid_line.append((x,y))
                    break
            else:
                grid_lines.append([(x,y)])

        # sort the rows by the x or y coordinate
        removable = []
        for idx, grid_line in enumerate(grid_lines):
            if len(grid_line) <= 9:
                removable.append(idx)
                continue
            grid_line.sort(key=lambda point: point[0] if by_rows else point[1])
        grid_lines = [grid_line for i, grid_line in enumerate(grid_lines) if i not in removable]
        return grid_lines

    def getSquares(self, image):
        # find contours on the preprocessed image
        img, contours, hierarchy = cv2.findContours(image.copy(),
                                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # sort the contours by the area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cont in contours:
            # calculate the perimeter and approximate the contour with a polygon
            perim = cv2.arcLength(cont, True)
            approx = cv2.approxPolyDP(cont, 0.04 * perim, True)
            # calculate the area
            contour_area = cv2.contourArea(cont)
            hull_area = cv2.contourArea(cv2.convexHull(cont))
            area_ratio = contour_area / float(hull_area)
            # if the approximation has 4 vertex and the area ratio is greater than 0.9 then
            # we assume that it's a square
            if len(approx) == 4 and area_ratio > 0.9:
                M = cv2.moments(cont)
                roi_img = RegionOfInterest(cont, M, approx)
                # the greatest square will be the sudoku table itself
                break
        else:
            # if there is no square
            roi_img = None
        return roi_img

    def getNumbers(self, grid_points):
        numbers = []
        if len(grid_points) == 100:
            for row in xrange(9):
                for col in xrange(9):
                    index = col + row * 10
                    corners = np.float32([[grid_points[index]],
                                          [grid_points[index + 1]],
                                          [grid_points[index + 11]],
                                          [grid_points[index + 10]]])
                    pers_matrix = cv2.getPerspectiveTransform(corners, self.number_corners)
                    square = cv2.warpPerspective(self.table_image, pers_matrix, (self.number_size, self.number_size))
                    square = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
                    _, square = cv2.threshold(square, 110, 255, cv2.THRESH_BINARY_INV)
                    # reshape numbers to get 28x28 pixels and crop the grid lines from the border
                    square = square[self.number_padding:self.number_size - self.number_padding,
                                    self.number_padding:self.number_size - self.number_padding]
                    input_img = np.ravel(square)
                    input_img = input_img.reshape((1, 784))
                    number_out = self.model.predict(input_img)
                    number_out = np.argmax(number_out[0])
                    numbers.append(Number(square, grid_points[index + 10], number_out))
        return numbers

if __name__ == '__main__':
    import time
    import sys

    def load_from_file(filename):
        input_img = cv2.imread(filename)

        start = time.time()
        processor = ImageProcessor()
        out_image = processor.new_image(input_img)
        end = time.time()
        print "Process time: %f" % (end-start)

        cv2.imshow('Table', out_image)
        cv2.waitKey()
        for idx, number in enumerate(processor.numbers):
            cv2.imwrite('numbers/numbers%d.jpg' % idx, number.image)

    def load_from_camera(camera_number):
        cap = cv2.VideoCapture(camera_number)

        if not cap.isOpened():
            print "Failed to open caption!"
            sys.exit(1)

        processor = ImageProcessor()

        while (True):
            # Capture frame-by-frame
            _, frame = cap.read()

            out_image = processor.new_image(frame)

            # Display the resulting frame
            cv2.imshow('frame', out_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                import glob
                cv2.imwrite('test_img%d.jpg' % len(glob.glob('*.jpg')), frame)

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

    # load_from_file('test_img3.jpg')
    load_from_camera(0)
