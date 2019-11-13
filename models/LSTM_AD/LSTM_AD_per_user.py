import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import random
from models import inject_frauds
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from datetime import timedelta, datetime
import time
from models import feature_engineering
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import TimeSeriesSplit
from keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score, \
                            average_precision_score, \
                            roc_auc_score, \
                            precision_recall_curve, \
                            auc, \
                            mean_absolute_error, \
                            confusion_matrix, \
                            accuracy_score, \
                            balanced_accuracy_score, \
                            fbeta_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import os

tscv = TimeSeriesSplit(n_splits=3)
# fix random seed for reproducibility
np.random.seed(7)

thirty_days = timedelta(30)
one_day = timedelta(1)
seven_days = timedelta(7)
two_weeks = timedelta(14)
one_hour = timedelta(0, 60 * 60)

# todo: ottimizzare il codice
def read_dataset():
    # reading the datasets
    bonifici = pd.read_csv("/home/mpapale/thesis/datasets/quiubi_bonifici.csv")
    segnalaz = pd.read_csv("/home/mpapale/thesis/datasets/bonifici_segnalaz.csv")
    bonifici.set_index('indice', inplace=True)
    segnalaz.set_index('indice', inplace=True)

    # dropping columns with useless data
    useless_features = ["CAP", "Servizio", "Status", "Paese", "Provincia", "Nazione", "IDTransazione", "CRO", "Causale",
                        "Valuta", "ProfSicurezza", "NumConto", "ABI", "CAB", "Intestatario", "Indirizzo"]
    bonifici = bonifici.drop(useless_features, axis=1)
    segnalaz = segnalaz.drop(useless_features, axis=1)
    # in future, try to use these features
    bonifici = bonifici.drop(["DataValuta", "DataEsecuzione", "Nominativo", "TipoOperazione"], axis=1)
    segnalaz = segnalaz.drop(["DataValuta", "DataEsecuzione", "Nominativo", "TipoOperazione"], axis=1)

    # datasets merge into bonifici
    bonifici.loc[:, "isFraud"] = np.zeros(len(bonifici.index))
    for index, row in segnalaz.iterrows():
        if index in bonifici.index:
            bonifici.loc[index, "isFraud"] = 1
        else:
            bonifici.append(row)
    bonifici.loc[:, "isFraud"] = pd.to_numeric(bonifici["isFraud"], downcast='integer')
    bonifici.Timestamp = pd.to_datetime(bonifici.Timestamp)
    return bonifici

def find_best_threshold(y_true, y_pred):
    probabilities = get_errors(y_true[:, 0:len(y_true[0]) - 1], y_pred)
    thresholds = np.linspace(min(probabilities), max(probabilities), num=200, endpoint=True)
    scores = {}
    for t in thresholds:
        labels = adjusted_classes(probabilities, t)
        f = fbeta_score(y_true[:,len(y_true[0]) - 1], labels, beta=0.1)
        scores[t] = f
    return max(scores, key=scores.get)

# This function adjusts class predictions based on the prediction threshold (t).
# todo: use builtin sklearn function
def adjusted_classes(y_scores, t):
    return [1 if y >= t else 0 for y in y_scores]

def evaluate(y_true, y_pred, threshold):
    roc_auc = roc_auc_score(y_true, y_pred)
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    precision_recall_auc = auc(recall, precision)
    average_precision = average_precision_score(y_true, y_pred)

    y_pred = adjusted_classes(y_pred, threshold)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    fbeta = fbeta_score(y_true, y_pred, beta=0.1)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)
    return f1, fbeta, accuracy, precision, recall, balanced_accuracy, confusion, roc_auc, precision_recall_auc, average_precision

def get_errors(y_true, y_pred):
    len_y = len(y_pred)
    errors = []
    for i in range(len_y):
        error = mean_absolute_error(y_true[i], y_pred[i])
        errors.append(error)
    return errors

def average(lst):
    return sum(lst) / len(lst)

def get_users_with_more_frauds(dataset_by_user):
    print("Getting users with more frauds...")
    users = []
    for user in dataset_by_user.groups.keys():
        # order by time
        group = dataset_by_user.get_group(user)
        if len(group[group["isFraud"] == 1]) > 5:
            print("user: ", user, " has: ", len(group), " transactions and ", group["isFraud"].sum(), "frauds.")
            users.append(user)
    return users

def get_users_with_more_transactions(dataset_by_user):
    users = []
    for user in dataset_by_user.groups.keys():
        # order by time
        group = dataset_by_user.get_group(user).sort_values(by='Timestamp', ascending=True)
        if len(group) > 100:
            print("user: ", user, ", len: ", len(group))
            if len(group) < 400:
                users.append(user)
    return users

# convert the dataset to LSTM input
def create_sequences(transactions, look_back=1):
    num_features = len(transactions.columns)
    dataX, dataY = [], []
    for i in range(len(transactions)-look_back-1):
        # do not consider the isFraud feature
        a = transactions.iloc[i:(i+look_back), 0: num_features - 1].values
        dataX.append(a)
        dataY.append(transactions.iloc[i + look_back, 0: num_features - 1].values)
    return np.array(dataX), np.array(dataY)

def create_balanced_sequences_with_no_frauds_in_history(dataset, look_back):
    cols = list(dataset.columns)
    cols.remove("isFraud")
    genuine_transactions = dataset.loc[dataset.isFraud == 0, cols]
    sequences = []
    next_transaction = []

    pointer = 0
    # NB: in look_back transactions there must not be present frauds
    while pointer < len(genuine_transactions.index) - look_back:
        sequence = genuine_transactions[pointer: pointer + look_back - 1]
        index_last_genuine_transaction = genuine_transactions.index[pointer + look_back - 1]

        # get the next transaction
        i = dataset.index.get_loc(index_last_genuine_transaction)
        try:
            sequence = sequence.append(dataset.iloc[i + 1][cols])
            next_transaction.append(dataset.iloc[i + 1][cols + ["isFraud"]])
            sequences.append(sequence.values)
        except Exception as e:
            print(e)
        pointer += 1
    return np.array(sequences), np.array(next_transaction)

def create_fit_model(x_train, y_train, look_back, num_features):
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(512, input_shape=(look_back, num_features)))
    model.add(Dense(num_features))
    model.compile(loss='mse', optimizer='adam')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    model.fit(x_train, y_train, epochs=100, verbose=0, validation_split=0.2, callbacks=[es])
    return model

# num_frauds_size means the ratio of the frauds that must be in test set
# num_genuine_size means the ratio of genuine that must be in test set
def val_test_split(x_val_test, y_val_test, num_frauds_size=0.5, num_genuine_size=0.5):
    x_no_frauds = x_val_test[np.where(y_val_test[:, len(y_val_test[0]) - 1] == 0)]
    y_no_frauds = y_val_test[np.where(y_val_test[:, len(y_val_test[0]) - 1] == 0)]
    x_frauds = x_val_test[np.where(y_val_test[:, len(y_val_test[0]) - 1] == 1)]
    y_frauds = y_val_test[np.where(y_val_test[:, len(y_val_test[0]) - 1] == 1)]

    frauds_pointer = int(len(y_frauds) * num_frauds_size)
    genuine_pointer = int(len(y_no_frauds) * num_genuine_size)

    list_indices_frauds = list(range(len(y_frauds)))
    list_indices_genuine = list(range(len(y_no_frauds)))

    frauds_indices = random.sample(list_indices_frauds, k=frauds_pointer)
    genuine_indices = random.sample(list_indices_genuine, k=genuine_pointer)
    frauds_indices_not_used = list(set(list_indices_frauds) - set(frauds_indices))
    genuine_indices_not_used = list(set(list_indices_genuine) - set(genuine_indices))

    x_val = np.concatenate([x_no_frauds[genuine_indices], x_frauds[frauds_indices]])
    x_test = np.concatenate([x_no_frauds[genuine_indices_not_used], x_frauds[frauds_indices_not_used]])
    y_val = np.concatenate([y_no_frauds[genuine_indices], y_frauds[frauds_indices]])
    y_test = np.concatenate([y_no_frauds[genuine_indices_not_used], y_frauds[frauds_indices_not_used]])

    return x_val, x_test, y_val, y_test

def rescale_features(dataset):
    for col in dataset.columns:
        dataset.loc[:, col] = minmax_scale(dataset[col], feature_range=(0, 1))
    return dataset

def train_model(x_train, y_train, look_back):
    x_train = x_train[:, len(x_train[0]) - look_back: , :]
    model = create_fit_model(x_train, y_train, look_back, len(x_train[0, 0, :]))
    train_score = model.evaluate(x_train, y_train, verbose=0)
    print("Look_back: ", look_back, "- Train average: ", train_score)
    return model

def train_model_with_cv(dataset_train, look_back):
    x, y = create_sequences(dataset_train, look_back)
    num_features = len(dataset_train.columns)
    train_scores = []
    test_scores = []

    for train_index, test_index in tscv.split(x):
        x_train, x_val_in_training = x[train_index], x[test_index]
        y_train, y_val_in_training = y[train_index], y[test_index]

        model = create_fit_model(x_train, y_train, look_back, num_features - 1)
        train_score = model.evaluate(x_train, y_train, verbose=0)
        test_score = model.evaluate(x_val_in_training, y_val_in_training, verbose=0)
        # print('Train Score: ', train_score, 'Test Score: ', test_score)
        train_scores.append(train_score)
        test_scores.append(test_score)

    avg_train_score = average(train_scores)
    avg_test_score = average(test_scores)

    print("Look_back: ", look_back, "- Train average: ", avg_train_score, "- Test average: ", avg_test_score)
    return model

def evaluate_model(model, dataset_val_test, look_back):
    x_val_test, y_val_test = create_balanced_sequences_with_no_frauds_in_history(dataset_val_test, look_back)

    confusion_matrices = []
    f1_scores = []
    average_accuracy_scores = []
    precision_scores = []
    recall_scores = []
    fbeta_scores = []
    accuracy_scores = []
    roc_auc_scores = []
    precision_recall_auc_scores = []
    average_precision_scores = []
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    print("-----Performance using lookback : ", look_back)
    times_to_repeat = 100
    for i in range(times_to_repeat):
        x_val, x_test, y_val, y_test = val_test_split(x_val_test, y_val_test, num_frauds_size=0.5, num_genuine_size=0.5)
        # print("Val frauds size :", y_val[:, len(y_val[0]) - 1].tolist().count(1), " - Test fruads size", y_test[:, len(y_test[0]) - 1].tolist().count(1))

        # using validation set to find the best threshold to divide frauds and genuine
        y_val_pred = model.predict(x_val)
        best_threshold = find_best_threshold(y_val, y_val_pred)

        y_test_pred = model.predict(x_test)
        probabilities = get_errors(y_test[:, 0:len(y_test[0]) - 1], y_test_pred)
        labels = adjusted_classes(probabilities, best_threshold)
        f1, fbeta, accuracy, precision, recall, average_accuracy, confusion, roc_auc, precision_recall_auc, average_precision = evaluate(
            y_test[:, len(y_test[0]) - 1], labels, best_threshold)

        # print(confusion)
        tp += confusion[0, 0]
        tn += confusion[1, 1]
        fn += confusion[0, 1]
        fp += confusion[1, 0]

        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
        average_accuracy_scores.append(average_accuracy)
        confusion_matrices.append(confusion)
        fbeta_scores.append(fbeta)
        accuracy_scores.append(accuracy)
        roc_auc_scores.append(roc_auc)
        precision_recall_auc_scores.append(precision_recall_auc)
        average_precision_scores.append(average_precision)

    print(np.array([np.array([tp / times_to_repeat, fn / times_to_repeat]), np.array([fp / times_to_repeat, tn / times_to_repeat])]))
    print("f1: ", average(f1_scores))
    print("precision: ", average(precision_scores))
    print("recall: ", average(recall_scores))
    print("average accuracy: ", average(average_accuracy_scores))
    print("fbeta_scores: ", average(fbeta_scores))
    print("accuracy: ", average(accuracy_scores))
    print("roc auc: ", average(roc_auc_scores))
    print("precision recall auc: ", average(precision_recall_auc_scores))
    print("average precision: ", average(average_precision_scores))


dataset = read_dataset()
dataset_by_user = dataset[dataset["isFraud"] == 0].reset_index(drop=True).groupby("UserID")
# users = get_users_with_more_frauds(dataset_by_user)
# users = get_users_with_more_transactions(dataset_by_user)

users = ["079fee0af27f1d2a688020b0dc34d9b4",
         "0f1a2f9e0118b7f7d391f84178b8893b",
         "2465e4faea2954eb10a09c7392c49ad5",
         "3c387bb14b369cdd3a86c65e6fda7db0",
         "3d01aa89ac96d0d0b73a42a68b1556e7",
         "45fc7689baf3f74ce186cbfef1c04533",
         "542aba60f688c1850f773b2f4f25f26d",
         "5ebe3a30b486e493d1f017fbfb9fd05c",
         "60b234fe5110937c1821e87e92b87a4b",
         "83e464735d321ad83eb1a2d242e67e00",
         "a2756a4678fff7a48e63a5921aff55c7",
         "d47928417bd21ab8df82ffd86b954149",
         "d88aa9fa459bfe9c8825798c16d0c5f8"]
# users = ["0f1a2f9e0118b7f7d391f84178b8893b"]
print("Considering ", len(users), " users.")

csv_list = os.listdir("./../dataset_engineered_per_user/")
for user in users:
    print("------------------------- Starting user: ", user)

    if user + "_train.csv" in csv_list:
        print("Dataset engineered found, using it.")
        dataset_train = pd.read_csv("./../dataset_engineered_per_user/" + user + "_train.csv")
        dataset_val_test = pd.read_csv("./../dataset_engineered_per_user/" + user + "_val_test.csv")
        # delete the header and the "indice"
        dataset = dataset.iloc[1:, 1:]
    else:
        print("Dataset engineered not found, creating it...")
        specific_user = dataset_by_user.get_group(user)
        specific_user = specific_user.sort_values(by="Timestamp")
        specific_user = specific_user.reset_index(drop=True)

        len_transactions = len(specific_user.index)
        dataset_train = specific_user.iloc[0: int(len_transactions * 50 / 100)]
        dataset_val_test = specific_user.iloc[int(len_transactions * 50 / 100):]

        dataset_val_test = inject_frauds.first_scenario(dataset_val_test, dataset)

        dataset_train = feature_engineering.feature_engineering(dataset_train)
        dataset_val_test = feature_engineering.feature_engineering(dataset_val_test)
        dataset_train.to_csv("./../dataset_engineered_per_user/" + user + "_train.csv")
        dataset_val_test.to_csv("./../dataset_engineered_per_user/" + user + "_val_test.csv")

    dataset_train = dataset_train.drop(['Timestamp'], axis=1)
    dataset_val_test = dataset_val_test.drop(['Timestamp'], axis=1)
    
    dataset_train = rescale_features(dataset_train)
    dataset_val_test = rescale_features(dataset_val_test)

    look_backs = [1, 10, 30, 50]
    # look_backs = [10]
    x_train, y_train = create_sequences(dataset_train, max(look_backs))

    for look_back in look_backs:
        # model = train_model_with_cv(dataset_train, look_back)
        model = train_model(x_train, y_train, look_back)
        evaluate_model(model, dataset_val_test, look_back)