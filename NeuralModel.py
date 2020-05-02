import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


class NeuralModel:
    def __init__(self, vocab_size):
        model = Sequential()
        model.add(LSTM(100, input_shape=(1, 100),return_sequences=True))
        model.add(Dense(100))
        model.compile(loss='mean_absolute_error', optimizer='adam',metrics=['accuracy'])
        model.fit(data, target, nb_epoch=1000, batch_size=1, verbose=2, validation_data=(x_test, y_test))

        self.model = model
