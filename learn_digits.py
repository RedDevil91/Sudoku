import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn import datasets
from sklearn.model_selection import train_test_split


# define baseline model
def baseline_model(num_pixels, num_classes):
    # create model
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, init='normal', activation='relu'))
    model.add(Dense(100, init='normal', activation='relu'))
    model.add(Dense(50, init='normal', activation='relu'))
    model.add(Dense(num_classes, init='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

dataset = datasets.fetch_mldata('MNIST Original')

X = dataset.data
Y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, Y)

# print X_train.shape, X_test.shape

# one hot encoding of labels
y_test = np_utils.to_categorical(y_test.astype(np.int32))
y_train = np_utils.to_categorical(y_train.astype(np.int32))
# normalize inputs
X_train = np.float64(X_train / 255.)
X_test = np.float64(X_test / 255.)

# build the model
model = baseline_model(784, 10)
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
model.save_weights("mnist_model_weights.h5")
model_json = model.to_json()
with open('mnist_model.json', 'w+') as json_file:
    json_file.write(model_json)
