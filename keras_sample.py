from keras.models import Sequential

from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adagrad
from sklearn.pipeline import FeatureUnion, Pipeline

import numpy as np

def keras_sample():

    x1 = [0.45, -1.47,  1.5, -0.43,  1.52, -0.99,  1.2, 0.22, -0.67]
    x2 = [0.23,  1.03, -1.5,  0.12, -2.53,  1.79, -1.2, 0.22,  0.63]
    x3  =[   0,     1,    2,     3,      4,    5,    6,    7,     8]

    train_x = [x1, x2, x3]
    train_y = [234,  2.34, 2340]

    keras_model = Sequential()

    keras_model.add(Dense(32, input_dim=9))
    # now the model will take as input arrays of shape (*, 9)
    # and output arrays of shape (*, 32)

    keras_model.add(Dense(32))
    #created middle layer

    keras_model.add(Dense(1))
    #created output layer we have output values as intergers then dimension is 1

    Y_np  = np.array(train_y)
    #Y_np = Y_np.reshape(Y_np.size, 1)
    X_np  = np.array(train_x)

    print("Compile Keras model")
    #keras_model.compile(loss='mse', optimizer=Adagrad(lr=0.1))
    keras_model.compile(loss='mse', optimizer='adam')
    print("Fit Keras model")
    keras_model.fit(X_np, Y_np, nb_epoch=10, batch_size=16)

    print("Finished")


def main():
    keras_sample()
    return

if __name__ == '__main__':
    main()
