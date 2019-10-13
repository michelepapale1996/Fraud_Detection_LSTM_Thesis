import time
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, Callback, TensorBoard, ModelCheckpoint
import keras.backend as K
import shutil
import os

class VanillaLSTM(object):
    # layers = {input: 1, 2: 64, 3: 256, 4: 100, output: 1}
    def __init__(self, look_back, layers, dropout, loss, learning_rate):
        self.look_back = look_back
        self.n_hidden = len(layers) - 2
        self.model = Sequential()
        self.layers = layers
        self.loss = loss
        # self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.dropout = dropout

    def build_model(self):
        # first add input to hidden1
        self.model.add(LSTM(
            self.layers['input'],
            input_shape=(self.look_back, self.layers['input']),
            return_sequences=True,
            ))
        self.model.add(Dropout(self.dropout))

        self.model.add(LSTM(self.layers['hidden1'], return_sequences=False))
        self.model.add(Dropout(self.dropout))

        # add output
        self.model.add(Dense(self.layers['output']))

        # compile model and print summary
        self.model.compile(loss=self.loss, optimizer=Adam(lr=self.learning_rate))
        self.model.summary()
        return self.model


class LstmClassifier(object):
    def __init__(self, look_back, layers, dropout, loss, learning_rate, num_features, metrics):
        self.look_back = look_back
        self.n_hidden = len(layers) - 2
        self.model = Sequential()
        self.layers = layers
        self.loss = loss
        # self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.num_features = num_features
        self.metrics = metrics

    def build_model(self):
        # first add input to hidden1
        self.model.add(LSTM(
            self.layers['input'],
            input_shape=(self.look_back + 1, self.num_features),
            return_sequences=True,
            ))

        self.model.add(LSTM(self.layers['hidden1'], return_sequences=False))

        # add output
        self.model.add(Dense(self.layers['output'], activation='sigmoid'))

        # compile model and print summary
        self.model.compile(loss=self.loss, optimizer=Adam(lr=self.learning_rate), metrics=self.metrics)
        # self.model.summary()
        return self.model


def train_model(model, x_train, y_train, batch_size, epochs, shuffle=False, validation=False, patience=0):
    print("Training...")
    training_start_time = time.time()

    if validation:
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
        history_callback = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=shuffle, verbose=2, callbacks=[early_stopping])
    else:
        history_callback = model.fit(x_train, y_train, epochs=epochs, verbose=2, callbacks=[], shuffle=shuffle)

    print('Training done. Duration: ' + str(time.time() - training_start_time))
    print("Training Loss per epoch: " + str(history_callback.history["loss"]))
    if validation:
        print("Validation  Loss per epoch: %s" % str(history_callback.history["val_loss"]))
    return history_callback
    # for epoch in range(epochs):
    #     model.fit(x_train, y_train, batch_size=batch_size, epochs=1,shuffle=shuffle, verbose=2)
    #     model.reset_states()