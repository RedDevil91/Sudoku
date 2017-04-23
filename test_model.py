import cv2
import numpy as np
import glob
from keras.models import model_from_json


def load_image(image_path):
    number_image = cv2.imread(image_path)
    number_image = cv2.cvtColor(number_image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Number', number_image)
    cv2.waitKey()
    number_image = np.ravel(number_image)
    number_image = number_image.reshape((1, 784))
    return number_image

with open('mnist_model.json', 'r+') as json_file:
    loaded_json_model = json_file.read()
model = model_from_json(loaded_json_model)
model.load_weights('mnist_model_weights.h5')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

image_list = glob.glob('./numbers/*.jpg')
for grid_square in image_list:
    test_image = load_image(grid_square)

    out = model.predict(test_image)
    out = np.argmax(out[0])
    print 'And the number is: %d' % out
