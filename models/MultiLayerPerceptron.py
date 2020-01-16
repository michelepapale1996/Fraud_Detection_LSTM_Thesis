import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from dataset_creation import sequences_crafting_for_classification, constants
from models import LSTM_classifier, evaluation
from sklearn.model_selection import RandomizedSearchCV
import warnings
sns.set(style="white", color_codes=True)
sns.set_context(rc={"font.family":'sans',"font.size":24,"axes.titlesize":24,"axes.labelsize":24})
# seaborn can generate several warnings, we ignore them
warnings.filterwarnings("ignore")

def create_model(x_train, y_train):
    model = Sequential()
    model.add(Dense(64, input_dim=len(x_train[0]), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.fit(x_train, y_train, epochs=30, verbose=1, batch_size=30)

    return model


look_back = constants.LOOK_BACK
# x_train, y_train = sequences_crafting_for_classification.create_train_set(look_back)
x_train = np.load("/home/mpapale/thesis/classification/dataset_" + str(look_back) + "_lookback/x_train_50_trx_per_user.npy")
y_train = np.load("/home/mpapale/thesis/classification/dataset_" + str(look_back) + "_lookback/y_train_50_trx_per_user.npy")
# x_test, y_test = sequences_crafting_for_classification.create_test_set(look_back)
x_test = np.load("/home/mpapale/thesis/classification/dataset_" + str(look_back) + "_lookback/x_test_50_trx_per_user.npy")
y_test = np.load("/home/mpapale/thesis/classification/dataset_" + str(look_back) + "_lookback/y_test_50_trx_per_user.npy")
# adapt train and test set to supervised learning without time windows
x_train_sup = x_train[:, look_back, :]
x_test_sup = x_test[:, look_back, :]

print("Fitting model...")
model = create_model(x_train_sup, y_train)

print("Evaluating model...")
y_pred = model.predict_proba(x_test_sup)
evaluation.evaluate_n_times(y_test, y_pred.ravel())