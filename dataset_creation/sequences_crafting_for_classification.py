import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
# seaborn can generate several warnings, we ignore them
import warnings
from datetime import timedelta
from dataset_creation import constants
import os
# in order to print all the columns
pd.set_option('display.max_columns', 100)
sns.set(style="white", color_codes=True)
sns.set_context(rc={"font.family": 'sans', "font.size": 24, "axes.titlesize": 24, "axes.labelsize": 24})
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------------------------
# ---------------------- Module used to create the dataset given to the LSTM --------------------
# -----------------------------------------------------------------------------------------------

# creates samples starting by the sequence of transactions of a given user
# NB: only the last transaction of a sequence can be a fraud
def get_sequences_from_user(transactions, look_back):
    sequences = []
    labels = []

    pointer = look_back + 1
    # for each transaction, check if there are lookback genuine transactions
    while pointer < len(transactions.index):
        past_transactions = transactions.iloc[: pointer]
        # lookback transactions must be genuine
        look_back_window = past_transactions[past_transactions.isFraud == 0]

        if len(look_back_window) >= look_back:
            look_back_window = look_back_window.tail(look_back)
            look_back_window = look_back_window.drop(["Timestamp", "UserID", "isFraud"], axis=1)
            current_trx = transactions.iloc[pointer:pointer + 1]
            labels.append(current_trx["isFraud"].values[0])
            current_trx = current_trx.drop(["Timestamp", "UserID", "isFraud"], axis=1)

            if look_back > 0:
                to_append = look_back_window.append(current_trx, ignore_index=True)
                sequences.append(to_append.values)
            else:
                sequences.append(current_trx.values[0])

        pointer += 1
    return sequences, labels


# converting low level dataset (transactions of users)
# to high level
# x will be: sequences of transactions with at least length = look_back + 1
# y will be the "isFraud" flag
def create_sequences(dataset, look_back=1):
    bonifici_by_user = dataset.groupby("UserID")
    # print("Creating samples using as look_back = " + str(look_back))
    # print("Total number of users: ", len(bonifici_by_user.groups.keys()))
    x = []
    y = []

    for user in bonifici_by_user.groups.keys():
        transactions = bonifici_by_user.get_group(user).sort_values("Timestamp").reset_index(drop=True)
        sequence_x_train, sequence_y_train = get_sequences_from_user(transactions, look_back)

        x.extend(sequence_x_train)
        y.extend(sequence_y_train)

    return np.asarray(x), np.asarray(y)

def insert_lookback_transactions_from_training_set(d_train, d_test, lookback):
    print("Inserting in test set lookback transactions from training...")
    dataset_by_user = d_test.groupby("UserID")
    for user in dataset_by_user.groups.keys():
        training_transactions = d_train[d_train.UserID == user].sort_values(by='Timestamp', ascending=True).reset_index(drop=True)
        # take only genuine transactions
        training_genuine_transactions = training_transactions[training_transactions.isFraud == 0].reset_index(drop=True)
        last_training_transactions = training_genuine_transactions.tail(lookback)
        group = dataset_by_user.get_group(user).sort_values(by='Timestamp', ascending=True).reset_index(drop=True)

        try:
            extended_dataset = extended_dataset.append(group, ignore_index=True)
        # if it is the first iteration, extended_dataset is not defined
        except NameError:
            extended_dataset = group

        extended_dataset = extended_dataset.append(last_training_transactions, ignore_index=True)

    return extended_dataset

def create_train_set(look_back, path):
    dataset_train = pd.read_csv(path, parse_dates=True)
    dataset_train = dataset_train.drop(["IP", "IBAN", "IBAN_CC", "CC_ASN"], axis=1)

    x_train, y_train = create_sequences(dataset_train, look_back)
    return x_train, y_train

def create_test_set(look_back, train_path, test_path):
    dataset_train = pd.read_csv(train_path, parse_dates=True)
    dataset_train = dataset_train.drop(["IP", "IBAN", "IBAN_CC", "CC_ASN"], axis=1)
    dataset_test = pd.read_csv(test_path, parse_dates=True)
    dataset_test = dataset_test.drop(["IP", "IBAN", "IBAN_CC", "CC_ASN"], axis=1)

    print("len test set before injecting training transactions", len(dataset_test))
    # in test set, insert lookback transactions for each user taking them from training set.
    # in this way, all the transactions in test set will be used.
    dataset_test = insert_lookback_transactions_from_training_set(dataset_train, dataset_test, look_back)
    print("len test set after injecting training transactions", len(dataset_test))
    x_test, y_test = create_sequences(dataset_test, look_back)
    return x_test, y_test

def get_file_name(dataset_type=constants.DATASET_TYPE, scenario_type=constants.ALL_SCENARIOS):
    if dataset_type == constants.INJECTED_DATASET:
        path = "test_696_users_" + scenario_type + "_scenario"
        train_path = "train_696_users_" + scenario_type + "_scenario"
        # path = "test_4072_users_" + scenario_type + "_scenario"
        # train_path = "train_4072_users_" + scenario_type + "_scenario"
    if dataset_type == constants.FRAUD_BUSTER_DATASET:
        path = "fraud_buster_test_250_users_" + scenario_type + "_scenario"
    if dataset_type == constants.REAL_DATASET:
        path = "real_dataset_test_696_users"
        train_path = "real_dataset_train_696_users"
        # path = "real_dataset_test_4072_users"
        # train_path = "real_dataset_train_4072_users"
    return train_path, path

# used from other models to get the train set (without recreate the sequences if they already exists)
def get_train_set(dataset_type=constants.DATASET_TYPE, scenario=constants.ALL_SCENARIOS):
    train_file_name, _ = get_file_name(dataset_type, scenario)
    x_train_path = "../classification/dataset_" + str(constants.LOOK_BACK) + "_lookback/x_" + train_file_name + ".npy"
    y_train_path = "../classification/dataset_" + str(constants.LOOK_BACK) + "_lookback/y_" + train_file_name + ".npy"
    print("Using as train set:", x_train_path)
    # check if train set already exist, otherwise create and save it
    if not os.path.exists(x_train_path):
        print("File does not exist, creating it...")
        main(dataset_type, scenario)

    x_train = np.load(x_train_path)
    y_train = np.load(y_train_path)
    return x_train, y_train

# used from other models to get the test set (without recreate the sequences if they already exists)
def get_test_set(dataset_type=constants.DATASET_TYPE, scenario=constants.ALL_SCENARIOS):
    _, test_file_name = get_file_name(dataset_type, scenario)
    x_test_path = "../classification/dataset_" + str(constants.LOOK_BACK) + "_lookback/x_" + test_file_name + ".npy"
    y_test_path = "../classification/dataset_" + str(constants.LOOK_BACK) + "_lookback/y_" + test_file_name + ".npy"
    print("Using as test set:", x_test_path)
    # check if train set already exist, otherwise create and save it
    if not os.path.exists(x_test_path):
        print("File does not exist, creating it...")
        main(dataset_type, scenario)

    x_test = np.load(x_test_path)
    y_test = np.load(y_test_path)
    return x_test, y_test

def main(dataset_type=constants.DATASET_TYPE, scenario=constants.ALL_SCENARIOS):
    look_back = constants.LOOK_BACK
    train_file_name, test_file_name = get_file_name(dataset_type, scenario)

    train_path = "../datasets/" + train_file_name + ".csv"
    test_path = "../datasets/" + test_file_name + ".csv"
    print("Creating train set: ", train_path)
    x_train, y_train = create_train_set(look_back, train_path)
    print("Creating test set: ", test_path)
    x_test, y_test = create_test_set(look_back, train_path, test_path)

    np.save("../classification/dataset_" + str(look_back) + "_lookback/x_" + train_file_name + ".npy", x_train)
    np.save("../classification/dataset_" + str(look_back) + "_lookback/y_" + train_file_name + ".npy", y_train)
    np.save("../classification/dataset_" + str(look_back) + "_lookback/x_" + test_file_name + ".npy", x_test)
    np.save("../classification/dataset_" + str(look_back) + "_lookback/y_" + test_file_name + ".npy", y_test)


if __name__ == "__main__":
    main()