import pandas as pd
import numpy as np
import random
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
# seaborn can generate several warnings, we ignore them
import warnings
from keras.wrappers import scikit_learn
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, PredefinedSplit
import sys
sys.path.append("/home/mpapale/thesis")
from models import resampling_dataset
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import EarlyStopping
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.optimizers import Adam, RMSprop
from keras.models import Sequential
from sklearn.datasets import make_classification
from dataset_creation import feature_engineering, sequences_crafting_for_classification, constants
from models import evaluation, explainability

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

def create_fit_model(x_train, y_train, look_back, params=None):
    if not params:
        if constants.USING_AGGREGATED_FEATURES:
            if constants.DATASET_TYPE == constants.INJECTED_DATASET:
                params = constants.BEST_PARAMS_LSTM_AGGREGATED
            if constants.DATASET_TYPE == constants.REAL_DATASET:
                params = constants.BEST_PARAMS_LSTM_REAL_DATASET_AGGREGATED
            if constants.DATASET_TYPE == constants.OLD_DATASET:
                params = constants.BEST_PARAMS_LSTM_OLD_DATASET_AGGREGATED
        else:
            if constants.DATASET_TYPE == constants.INJECTED_DATASET:
                params = constants.BEST_PARAMS_LSTM_NO_AGGREGATED
            if constants.DATASET_TYPE == constants.REAL_DATASET:
                params = constants.BEST_PARAMS_LSTM_REAL_DATASET_NO_AGGREGATED
            if constants.DATASET_TYPE == constants.OLD_DATASET:
                params = constants.BEST_PARAMS_LSTM_OLD_DATASET_NO_AGGREGATED

    # model = create_simple_model(x_train, y_train, look_back)
    model = create_model(params["layers"], params["dropout_rate"], look_back, len(x_train[0, 0]))
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    # model.fit(x_train, y_train, epochs=20, verbose=1, validation_split=0.2, shuffle=False, callbacks=[es])
    model.fit(x_train, y_train, epochs=params["epochs"], verbose=1, batch_size=params["batch_size"])
    return model

def create_simple_model(x_train, y_train, look_back):
    print("Creating the lstm model...")
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(96, input_shape=(look_back + 1, len(x_train[0, 0])), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(64, input_shape=(look_back + 1, len(x_train[0, 0])), return_sequences=False))
    model.add(Dropout(0.3))
    #model.add(LSTM(128, input_shape=(look_back + 1, len(x_train[0, 0])), return_sequences=False))
    #model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
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


def create_model(layers, dropout_rate, look_back, num_features):
    model = Sequential()
    # -2 because there are the input and the output layer
    # +1 because range is 0-indexed
    n_hidden = len(layers) - 2
    if n_hidden > 0:
        model.add(LSTM(layers['input'], input_shape=(look_back+1, num_features), return_sequences=False))
        model.add(Dropout(dropout_rate))
        # add hidden layers return sequence true
        for i in range(1, n_hidden):
            model.add(Dense(layers["hidden" + str(i)]))
            model.add(Dropout(dropout_rate))
        # add hidden_last return Sequences False
        model.add(Dense(layers['hidden' + str(n_hidden)]))
        model.add(Dropout(dropout_rate))
    else:
        model.add(LSTM(layers['input'], input_shape=(look_back+1, num_features), return_sequences=False))
        model.add(Dropout(dropout_rate))

    model.add(Dense(layers['output'], activation='sigmoid'))
    model.compile(loss="mse", optimizer="adam")
    # model.summary()
    return model


def model_selection(x, y, look_back):
    # define the grid search parameters
    batch_size = [16, 64, 128]
    epochs = [10, 25, 50]
    layers = [{'input': 64, 'output': 1},
              {'input': 128, 'output': 1},
              {'input': 256, 'output': 1},
              {'input': 128, 'hidden1': 64, 'output': 1},
              {'input': 256, 'hidden1': 64, 'hidden2': 256, 'output': 1}]
    dropout_rate = [0.2, 0.5, 0.8]

    # batch_size = np.random.choice(batch_size, int(len(batch_size) / 2))
    # epochs = np.random.choice(epochs, int(len(epochs) / 2))
    # layers = np.random.choice(layers, int(len(layers) / 2))
    # learning_rate = np.random.choice(learning_rate, int(len(learning_rate) / 2))
    # dropout_rate = np.random.choice(dropout_rate, int(len(dropout_rate) / 2))

    model = scikit_learn.KerasClassifier(build_fn=create_model, look_back=look_back, num_features=len(x[0, 0, :]), verbose=0)
    param_grid = dict(layers=layers, batch_size=batch_size, epochs=epochs, dropout_rate=dropout_rate)

    '''
    validation_fold = [-1 for _ in range(len(x_train))] + [0 for _ in range(len(x_val))]
    x_train = np.append(x_train, x_val, axis=0)
    y_train = np.append(y_train, y_val, axis=0)
    ps = PredefinedSplit(validation_fold)
    grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100, n_jobs=-1, cv=ps, scoring="roc_auc", verbose=1)
    '''

    # grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=50, n_jobs=-1, cv=3, scoring="roc_auc", verbose=0)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, scoring="roc_auc", verbose=3)

    grid_result = grid.fit(x, y)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    return grid_result


if __name__ == "__main__":
    look_back = constants.LOOK_BACK
    print("Lookback using: ", look_back)
    x_train, y_train = sequences_crafting_for_classification.get_train_set()

    # if the dataset is the real one -> contrast imbalanced dataset problem
    if constants.DATASET_TYPE == constants.REAL_DATASET:
        x_train, y_train = resampling_dataset.oversample_set(x_train, y_train)

    x_test, y_test = sequences_crafting_for_classification.get_test_set()
    '''
    model = create_fit_model(x_train, y_train, look_back)
    testPredict = model.predict(x_test)
    evaluation.evaluate_n_times(y_test, testPredict)
    explainability.explain_dataset(model, x_test, y_test)
    '''

    model = model_selection(x_train, y_train, look_back)
    y_pred = model.predict_proba(x_test)
    evaluation.evaluate_n_times(y_test, y_pred[:, 1])
    # runtime_testing_fraud_buster(model)
