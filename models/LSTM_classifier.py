import pandas as pd
import numpy as np
import random
import seaborn as sns
# seaborn can generate several warnings, we ignore them
import warnings
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV

from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import EarlyStopping
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.optimizers import Adam, RMSprop
from keras.models import Sequential
from sklearn.datasets import make_classification
from dataset_creation import feature_engineering, sequences_crafting_for_classification, constants
from models import evaluation

from sklearn.model_selection import train_test_split
# in order to print all the columns
pd.set_option('display.max_columns', 100)
sns.set(style="white", color_codes=True)
sns.set_context(rc={"font.family": 'sans', "font.size": 24, "axes.titlesize": 24, "axes.labelsize": 24})
warnings.filterwarnings("ignore")

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# -----------------------------------------------------------------------------------------------
# --------------------------Main idea: creating an LSTM classifier ------------------------------
# -----------------------------------------------------------------------------------------------

def create_fit_model(x_train, y_train, look_back):
    print("Fitting the model...")
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(512, input_shape=(look_back + 1, len(x_train[0, 0])), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256, input_shape=(look_back + 1, len(x_train[0, 0])), return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    # model.fit(x_train, y_train, epochs=20, verbose=0, validation_split=0.2, shuffle=False, callbacks=[es])
    model.fit(x_train, y_train, epochs=5, verbose=1, batch_size=30)
    return model

def get_key(val, my_dict):
    for key, value in my_dict.items():
         if val == value:
             return key

def runtime_testing_fraud_buster(model):
    scenario_types = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
    for scenario in scenario_types:
        print("------------ ", scenario, "scenario ----------")
        dataset_train = pd.read_csv("/home/mpapale/thesis/datasets/fraud_buster_bonifici_engineered_train_250_users_" + scenario + "_scenario.csv", parse_dates=True)
        dataset_test = pd.read_csv("/home/mpapale/thesis/datasets/fraud_buster_bonifici_engineered_test_250_users_" + scenario + "_scenario.csv", parse_dates=True)
        dataset_train = dataset_train.drop(["IP", "IBAN", "IBAN_CC", "CC_ASN", "ASN"], axis=1)
        dataset_test = dataset_test.drop(["IP", "IBAN", "IBAN_CC", "CC_ASN", "ASN"], axis=1)
        dataset_by_user = dataset_test.groupby("UserID")

        is_defrauded = {}
        max_fraud_score = {}

        users = list(dataset_by_user.groups.keys())
        for user in users:
            user_transactions = dataset_train[dataset_train.UserID == user]
            user_transactions = user_transactions.append(dataset_test[dataset_test.UserID == user])
            user_transactions = user_transactions.sort_values(by="Timestamp")
            user_transactions.tail(len(dataset_test[dataset_test.UserID == user]) + look_back)

            x_test, y_test = sequences_crafting_for_classification.create_sequences(user_transactions, look_back)
            testPredict = model.predict(x_test)
            max_fraud_score[user] = max(np.reshape(testPredict, len(testPredict)))
            is_defrauded[user] = max(y_test)

        scores = list(max_fraud_score.values())
        scores.sort(reverse=True)
        num_users_to_infect = int(len(users) / 100 * 5)
        scores = scores[:5 * num_users_to_infect]
        my_tp = 0
        for score in scores:
            user = get_key(score, max_fraud_score)
            if is_defrauded[user] == 1:
                my_tp += 1

        print("TP: ", my_tp)


def create_model(layers, learning_rate, dropout_rate, look_back, num_features):
    model = Sequential()
    n_hidden = len(layers) - 2
    if n_hidden > 2:
        model.add(LSTM(layers['input'], input_shape=(look_back+1, num_features), return_sequences=True))

        # add hidden layers return sequence true
        for i in range(2, n_hidden):
            model.add(LSTM(layers["hidden" + str(i)], return_sequences=True))
            model.add(Dropout(dropout_rate))
        # add hidden_last return Sequences False
        model.add(LSTM(layers['hidden' + str(n_hidden)], return_sequences=False))
        model.add(Dropout(dropout_rate))
    else:
        model.add(LSTM(layers['input'], input_shape=(look_back+1, num_features), return_sequences=False))

    model.add(Dense(layers['output'], activation='sigmoid'))

    # model.compile(loss="mse", optimizer=Adam(lr=learning_rate))
    model.compile(loss="mse", optimizer="adam")
    # model.summary()
    return model


def model_selection(x_train, y_train, x_test, y_test, look_back):
    # define the grid search parameters
    batch_size = [10, 30, 50]
    epochs = [10, 25, 50]
    layers = [{'input': 256, 'hidden1': 64, 'output': 1},
              {'input': 512, 'hidden1': 256, 'output': 1},
              {'input': 512, 'hidden1': 256, 'hidden2': 64, 'output': 1}]
    learning_rate = [0.1, 0.01, 0.001]
    dropout_rate = [0.0, 0.3, 0.8]

    batch_size = np.random.choice(batch_size, int(len(batch_size) / 2))
    # epochs = np.random.choice(epochs, int(len(epochs) / 2))
    # layers = np.random.choice(layers, int(len(layers) / 2))
    learning_rate = np.random.choice(learning_rate, int(len(learning_rate) / 2))
    dropout_rate = np.random.choice(dropout_rate, int(len(dropout_rate) / 2))

    max_roc = 0
    best_model = False
    best_params = False

    for b in batch_size:
        for e in epochs:
            for l in layers:
                for lr in learning_rate:
                    for d in dropout_rate:

                        print("Current batch_size", b, "of", batch_size)
                        print("Current epochs", e, "of", epochs)
                        print("Current layers", l, "of", layers)
                        print("Current learning_rate", lr, "of", learning_rate)
                        print("Current dropout_rate", d, "of", dropout_rate)

                        model = create_model(l, lr, d, look_back, len(x_train[0, 0]))
                        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=int(e / 3))
                        # model.fit(x_train, y_train, epochs=e, verbose=1, validation_split=0.2, batch_size=b, shuffle=False, callbacks=[es])
                        model.fit(x_train, y_train, epochs=e, batch_size=b, verbose=1)
                        testPredict = model.predict(x_test)
                        f1, roc = evaluation.evaluate(y_test, testPredict)
                        if roc > max_roc:
                            best_model = model
                            best_params = {
                                "batch_size": b,
                                "epochs": e,
                                "layers": l,
                                "learning_rate": lr,
                                "dropout_rate": d
                            }
                            max_roc = roc
    return best_model, best_params


if __name__ == "__main__":
    look_back = constants.LOOK_BACK
    print("Lookback using: ", look_back)
    x_train, y_train = sequences_crafting_for_classification.create_train_set(look_back)
    x_test, y_test = sequences_crafting_for_classification.create_test_set(look_back)

    model = create_fit_model(x_train, y_train, look_back)
    # model, params = model_selection(x_train, y_train, x_test, y_test, look_back)
    # print("Best params found: ", params)
    testPredict = model.predict(x_test)
    evaluation.evaluate(y_test, testPredict)
    # runtime_testing_fraud_buster(model)