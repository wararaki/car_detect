'''
train
'''

import sys
import gc
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils

def normalize(x_data, y_data, n_class):
    '''
    normalize
    '''
    x_data = x_data.astype('float') / 256
    y_data = np_utils.to_categorical(y_data, n_class)
    return (x_data, y_data)

def load_train_data(data_path, n_class):
    '''
    load training data
    '''
    x_train, x_test, y_train, y_test = np.load(data_path)
    x_train, y_train = normalize(x_train, y_train, n_class)
    x_test, y_test = normalize(x_test, y_test, n_class)
    return (x_train, x_test, y_train, y_test)

def create_model(n_input, n_class):
    '''
    create cnn model
    '''
    model = Sequential()

    # input layer
    model.add(Conv2D(16, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=n_input))
    model.add(Conv2D(32, (3, 3),
              activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_class, activation='softmax'))

    # compile
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    return model

def main():
    '''
    main function
    '''
    train_data_path = './dataset/train.dat.npy'
    n_class = 2
    # load dataset
    x_train, x_test, y_train, y_test = load_train_data(train_data_path, n_class)

    # create cnn model
    model = create_model(x_train.shape[1:], n_class)

    # fit model
    model.fit(x_train, y_train, batch_size=4, nb_epoch=7)

    # save model
    model_path = './model/detect_model.hdf5'
    model.save(model_path)

    # evaluate
    score = model.evaluate(x_test, y_test)
    print('loss=', score[0])
    print('accuracy=', score[1])

    gc.collect()
    return 0


if __name__ == '__main__':
    sys.exit(main())
