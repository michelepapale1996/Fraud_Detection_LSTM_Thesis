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
    #layers = {input: 1, 2: 64, 3: 256, 4: 100, output: 1}
    def __init__(self, look_back, layers, dropout, loss, learning_rate):
        self.look_back = look_back
        self.n_hidden = len(layers) - 2
        self.model = Sequential()
        self.layers = layers
        self.loss = loss
        #self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.dropout = dropout

    def build_model(self):
        #first add input to hidden1
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

        #compile model and print summary
        self.model.compile(loss=self.loss, optimizer=Adam(lr=self.learning_rate))
        self.model.summary()
        return self.model


def train_model(model, x_train, y_train, batch_size, epochs, shuffle, validation, validation_data, patience):
    print("Training")
    training_start_time = time.time()
    # tensorboard = TensorBoard(log_dir='/home/esihakh/projects/github_thesis/lstm_anomaly_thesis/logs/tf_logs/', histogram_freq=5, write_graph=True, write_images=False)
    if validation:
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience)

        history_callback = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                                     validation_data=validation_data,
                                     shuffle=shuffle, verbose=2, callbacks=[early_stopping])
    else:
        history_callback = model.fit(x_train, y_train, epochs=epochs, verbose=2)
    print('Training duration (s) : %s', str(time.time() - training_start_time))
    print("Training Loss per epoch: %s" % str(history_callback.history["loss"]))
    if validation:
        print("Validation  Loss per epoch: %s" % str(history_callback.history["val_loss"]))
    print(history_callback.history.keys())
    return history_callback
    # for epoch in range(epochs):
    #     model.fit(x_train, y_train, batch_size=batch_size, epochs=1,shuffle=shuffle, verbose=2)
    #     model.reset_states()