import cv2
import numpy as np

from keras.models import model_from_json

test_image = cv2.imread('./numbers/numbers9.jpg')
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Number', test_image)
cv2.waitKey()
test_image = np.ravel(test_image)
test_image = test_image.reshape((1, 784))

with open('mnist_model.json', 'r+') as json_file:
    loaded_json_model = json_file.read()
model = model_from_json(loaded_json_model)
model.load_weights('mnist_model_weights.h5')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
out = model.predict(test_image)
print out
