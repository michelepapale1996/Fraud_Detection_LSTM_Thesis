import time
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, Callback, TensorBoard, ModelCheckpoint
import shutil
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, roc_curve, roc_auc_score, precision_recall_curve, auc, confusion_matrix, accuracy_score


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
        if self.n_hidden > 2:
            self.model.add(LSTM(
                self.layers['input'],
                input_shape=(self.look_back + 1, self.num_features),
                return_sequences=True,
            ))

            # add hidden layers return sequence true
            for i in range(2, self.n_hidden):
                self.model.add(LSTM(
                    self.layers["hidden" + str(i)],
                    return_sequences=True))
                self.model.add(Dropout(self.dropout))

            # add hidden_last return Sequences False
            self.model.add(LSTM(
                self.layers['hidden' + str(self.n_hidden)],
                return_sequences=False))
            self.model.add(Dropout(self.dropout))
        else:
            self.model.add(LSTM(
                self.layers['input'],
                input_shape=(self.look_back + 1, self.num_features),
                return_sequences=False,
            ))

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
        history_callback = model.fit(x_train,
                                     y_train,
                                     batch_size=batch_size,
                                     epochs=epochs,
                                     shuffle=shuffle,
                                     validation_split=0.33,
                                     verbose=1,
                                     callbacks=[early_stopping])
    else:
        history_callback = model.fit(x_train, y_train, epochs=epochs, verbose=0, callbacks=[], shuffle=shuffle)

    print('Training done. Duration: ' + str(time.time() - training_start_time))
    # print("Training Loss per epoch: " + str(history_callback.history["loss"]))
    if validation:
        plt.plot(history_callback.history['loss'])
        plt.plot(history_callback.history['val_loss'])
        plt.title('model train vs validation loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()
    return history_callback
    # for epoch in range(epochs):
    #     model.fit(x_train, y_train, batch_size=batch_size, epochs=1,shuffle=shuffle, verbose=2)
    #     model.reset_states()


def evaluate_performance(y_true, y_pred):
    print("---- Performance ----")
    roc_auc = roc_auc_score(y_true, y_pred)
    print("Roc_auc score: %.3f" % roc_auc)

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    precision_recall_auc = auc(recall, precision)
    print("Precision_recall_auc score: %.3f" % precision_recall_auc)

    def get_position(recall):
        for i in range(0, len(recall)):
            if recall[i] < 0.2:
                return i - 1

    # This function adjusts class predictions based on the prediction threshold (t).
    def adjusted_classes(y_scores, t):
        return [1 if y >= t else 0 for y in y_scores]

    position = get_position(recall)
    print("Precision: " + str(precision[position]))
    threshold = thresholds[position]
    y_pred = adjusted_classes(y_pred, threshold)
    f1 = f1_score(y_true, y_pred)
    print("f1 score: %.3f" % f1)
    print("accuracy: " + str(accuracy_score(y_true, y_pred)))
    print(confusion_matrix(y_true, y_pred))

    return f1