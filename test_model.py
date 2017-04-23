import cv2
import numpy as np
import glob
from keras.models import model_from_json

with open('mnist_model.json', 'r+') as json_file:
    loaded_json_model = json_file.read()
model = model_from_json(loaded_json_model)
model.load_weights('mnist_model_weights.h5')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

images = glob.glob('./numbers/*.jpg')

for grid_square in images:
    test_image = cv2.imread(grid_square)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Number', test_image)
    cv2.waitKey()
    test_image = np.ravel(test_image)
    test_image = test_image.reshape((1, 784))

    out = model.predict(test_image)
    out = np.argmax(out[0])
    print 'And the number is: %d' % out
