import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sys
sys.path.append("/home/mpapale/thesis")
import random
from models import inject_frauds
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from datetime import timedelta
from sklearn.model_selection import TimeSeriesSplit
from keras.callbacks import EarlyStopping
from models.LSTM_AD import mae_evaluation, feature_engineering

tscv = TimeSeriesSplit(n_splits=3)
# fix random seed for reproducibility
np.random.seed(7)

thirty_days = timedelta(30)
one_day = timedelta(1)
seven_days = timedelta(7)
two_weeks = timedelta(14)
one_hour = timedelta(0, 60 * 60)

def average(lst):
    return sum(lst) / len(lst)

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
    num_features = transactions.shape[1]
    dataX, dataY = [], []
    for i in range(transactions.shape[0] - look_back - 1):
        # do not consider the isFraud feature
        a = transactions[i:(i+look_back), 0: num_features]
        dataX.append(a)
        dataY.append(transactions[i + look_back, 0: num_features])
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
        sequence = genuine_transactions.iloc[pointer: pointer + look_back]
        index_last_genuine_transaction = genuine_transactions.index[pointer + look_back - 1]
        # get the next transaction
        index_next_transaction = index_last_genuine_transaction + 1

        next_transaction.append(dataset.iloc[index_next_transaction])
        sequences.append(sequence.values)
        pointer += 1
    return np.array(sequences), np.array(next_transaction)


def create_fit_model(x_train, y_train, look_back, num_features):
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(512, input_shape=(look_back, num_features), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64, input_shape=(look_back, num_features), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512, input_shape=(look_back, num_features)))
    model.add(Dropout(0.2))
    model.add(Dense(num_features))
    model.compile(loss='mse', optimizer='adam')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    model.fit(x_train, y_train, epochs=200, verbose=0, validation_split=0.2, shuffle=False, callbacks=[es])
    # model.fit(x_train, y_train, epochs=100, verbose=0)
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
    # genuine_indices = random.sample(list_indices_genuine, k=frauds_pointer)
    genuine_indices = random.sample(list_indices_genuine, k=genuine_pointer)
    frauds_indices_not_used = list(set(list_indices_frauds) - set(frauds_indices))
    genuine_indices_not_used = list(set(list_indices_genuine) - set(genuine_indices))

    x_val = np.concatenate([x_no_frauds[genuine_indices], x_frauds[frauds_indices]])
    x_test = np.concatenate([x_no_frauds[genuine_indices_not_used], x_frauds[frauds_indices_not_used]])
    y_val = np.concatenate([y_no_frauds[genuine_indices], y_frauds[frauds_indices]])
    y_test = np.concatenate([y_no_frauds[genuine_indices_not_used], y_frauds[frauds_indices_not_used]])

    return x_val, x_test, y_val, y_test

def train_model(x_train, y_train, look_back):
    model = create_fit_model(x_train, y_train, look_back, len(x_train[0, 0, :]))
    train_score = model.evaluate(x_train, y_train, verbose=0)
    print("Look_back: ", look_back, "- Train average: ", train_score)
    return model


def evaluate_model(model, x_val_test, y_val_test):
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

    times_to_repeat = 50
    for i in range(times_to_repeat):
        x_val, x_test, y_val, y_test = val_test_split(x_val_test, y_val_test, num_frauds_size=0.5, num_genuine_size=0.5)

        f1, fbeta, accuracy, precision, recall, average_accuracy, confusion, roc_auc, precision_recall_auc, average_precision = mae_evaluation.evaluate_model(
            model, x_val, y_val, x_test, y_test)
        # f1, fbeta, accuracy, precision, recall, average_accuracy, confusion = rf_evaluation.evaluate_model(model, x_val, y_val, x_test, y_test)

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
        # roc_auc_scores.append(roc_auc)
        # precision_recall_auc_scores.append(precision_recall_auc)
        # average_precision_scores.append(average_precision)

    print(np.array([np.array([tp / times_to_repeat, fn / times_to_repeat]),
                    np.array([fp / times_to_repeat, tn / times_to_repeat])]))
    print("f1: ", average(f1_scores))
    print("precision: ", average(precision_scores))
    print("recall: ", average(recall_scores))
    print("average accuracy: ", average(average_accuracy_scores))
    print("fbeta_scores: ", average(fbeta_scores))
    print("accuracy: ", average(accuracy_scores))
    # print("roc auc: ", average(roc_auc_scores))
    # print("precision recall auc: ", average(precision_recall_auc_scores))
    # print("average precision: ", average(average_precision_scores))

def get_number_user_transactions_per_day(specific_user):
    trx_user = specific_user.sort_values("Timestamp")

    def daterange(start_date, end_date):
        for n in range(int ((end_date - start_date).days)):
            yield start_date + timedelta(n)

    start_date = trx_user["Timestamp"].iloc[0]
    end_date = trx_user["Timestamp"].iloc[len(trx_user.index.values) - 1]
    transactions_per_day = []
    for single_date in daterange(start_date, end_date):
        num_transactions_in_day = 0
        for i in range(len(trx_user["Timestamp"].index.values)):
            if trx_user["Timestamp"].iloc[i].date() == single_date:
                # print("found: ", single_date.strftime("%Y-%m-%d"), ", ", trx_user["Timestamp"].iloc[i].date())
                num_transactions_in_day += 1
        transactions_per_day.append(num_transactions_in_day)
    return transactions_per_day

def get_mean_amount_per_day(specific_user):
    trx_user = specific_user.sort_values("Timestamp")

    def daterange(start_date, end_date):
        for n in range(int ((end_date - start_date).days)):
            yield start_date + timedelta(n)

    start_date = trx_user["Timestamp"].iloc[0]
    end_date = trx_user["Timestamp"].iloc[len(trx_user.index.values) - 1]
    amounts_per_day = []
    for single_date in daterange(start_date, end_date):
        num_transactions_in_day = 0
        amount_in_day = 0
        for i in range(len(trx_user["Timestamp"].index.values)):
            if trx_user["Timestamp"].iloc[i].date() == single_date:
                # print("found: ", single_date.strftime("%Y-%m-%d"), ", ", trx_user["Timestamp"].iloc[i].date())
                amount_in_day += trx_user["Importo"].iloc[i]
                num_transactions_in_day += 1
        if num_transactions_in_day == 0:
            amounts_per_day.append(0)
        else:
            amounts_per_day.append(amount_in_day / num_transactions_in_day)
    return amounts_per_day


original_dataset = read_dataset()
dataset_by_user = original_dataset[original_dataset["isFraud"] == 0].reset_index(drop=True).groupby("UserID")
# users = get_users_with_more_frauds(dataset_by_user)
# users = get_users_with_more_transactions(dataset_by_user)
users = ["3c809ee390f8d8d200147cf114773110"]
look_backs = [30, 20, 15, 10, 5]
# look_backs = [30]

for user in users:
    print("------------------------- Starting user: ", user)
    for look_back in look_backs:
        specific_user = dataset_by_user.get_group(user)
        specific_user = specific_user.sort_values(by="Timestamp")
        specific_user = specific_user.reset_index(drop=True)

        len_trasanctions = len(specific_user.index)
        last_index_train = len_trasanctions - look_back - 20
        last_index_val = len_trasanctions - look_back - 10

        specific_user = inject_frauds.first_scenario(specific_user, original_dataset, last_index_train, last_index_val)
        specific_user = feature_engineering.feature_engineering(specific_user)

        # specific_user = specific_user[["Importo", "time_delta", "isFraud"]]
        specific_user = specific_user["Importo"]
        # specific_user = pd.DataFrame(get_mean_amount_per_day(specific_user))
        specific_user = pd.DataFrame(np.diff(specific_user))

        '''
        # removing outliers
        threshold = mean(specific_user.iloc[:, 0]) + stdev(specific_user.iloc[:, 0])
        print("Removing ", len(specific_user[specific_user.iloc[:, 0] > threshold]), "outliers")
        specific_user = specific_user[specific_user.iloc[:, 0] <= threshold]
        '''

        dataset_train = specific_user.iloc[0: last_index_train + 10]
        # dataset_val = specific_user.iloc[last_index_train: last_index_val]
        dataset_test = specific_user.iloc[last_index_val:]

        '''
        train = {#"time_delta": np.diff(dataset_train.time_delta),
                 "Importo": np.diff(dataset_train.Importo),
                 "isFraud": dataset_train.isFraud.iloc[1:]}
    
        test = {#"time_delta": np.diff(dataset_test.time_delta),
                "Importo": np.diff(dataset_test.Importo),
                "isFraud": dataset_test.isFraud.iloc[1:]}
    
        dataset_train = pd.DataFrame(data=train)
        dataset_test = pd.DataFrame(data=test)
        '''

        scaler = MinMaxScaler()
        dataset_train = scaler.fit_transform(dataset_train)
        #dataset_val = scaler.transform(dataset_val)
        dataset_test = scaler.transform(dataset_test)
        #dataset = np.concatenate((dataset_train, dataset_val, dataset_test), axis=0)
        dataset = np.concatenate((dataset_train, dataset_test), axis=0)

        '''    
        dataset_train.to_csv("./../dataset_engineered_per_user/" + user + "_train.csv")
        dataset_val_test.to_csv("./../dataset_engineered_per_user/" + user + "_val_test.csv")
        '''

        # x_train, y_train = create_sequences(dataset_train, max(look_backs))
        # x_val_test, y_val_test = create_sequences(dataset_test, max(look_backs))
        # x_val_test, y_val_test = create_balanced_sequences_with_no_frauds_in_history(dataset_val_test, max(look_backs))


        x, y = create_sequences(dataset, look_back)
        x_train, y_train = x[:last_index_train], y[:last_index_train]

        # do not consider "isFraud"
        # x_train = x_train[:, :, 0:2]
        # y_train = y_train[:, 0:2]

        # x_val, y_val = x[last_index_train:last_index_val], y[last_index_train:last_index_val]
        x_test, y_test = x[last_index_val:], y[last_index_val:]

        if len(x_train) == 0 or len(x_test) == 0:
            continue

        print("-----Performance using lookback : ", look_back)
        model = train_model(x_train, y_train, look_back)


        trainPredict = model.predict(x_train)
        testPredict = model.predict(x_test)
        '''
        amount_lstm = mean_absolute_error(y_val_test[:, 0], testPredict[:, 0])
        amount_mean = mean_absolute_error(y_val_test[:, 0], [average(y_train[:, 0])] * len(y_val_test))

        print("MAE on timedelta_diff using mean: ", amount_mean)
        print("MAE on timedelta_diff using LSTM: ", amount_lstm)
        '''
        plt.plot(trainPredict, label="train_predicted")
        plt.plot(y_train, label="y_train")
        plt.plot(np.concatenate(([np.nan] * len(y_train), np.reshape(y_test, len(y_test))), axis=0), label="y_test")
        plt.plot(np.concatenate(([np.nan] * len(y_train), np.reshape(testPredict, len(testPredict))), axis=0), label="test_predicted")
        plt.legend()
        plt.title("User timedelta_diff")
        plt.savefig("AAAuser_" + user + "timedelta_diff" + str(look_back) + ".png")
        plt.show()

        '''
        f1, fbeta, accuracy, precision, recall, average_accuracy, confusion, roc_auc, precision_recall_auc, average_precision = mae_evaluation.evaluate_model(
            model, x_val, y_val, x_test, y_test)

        print("f1: ", f1)
        print("precision: ", precision)
        print("recall: ", recall)
        print("average accuracy: ", average_accuracy)
        print("fbeta_scores: ", fbeta)
        print("accuracy: ", accuracy)
        print(confusion)
        '''