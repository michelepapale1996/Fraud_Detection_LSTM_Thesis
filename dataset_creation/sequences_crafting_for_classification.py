import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
# seaborn can generate several warnings, we ignore them
import warnings
from datetime import timedelta
from dataset_creation import constants

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

            if len(look_back_window) >= 1:
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

    number_of_users = 0
    counter = 0
    for user in bonifici_by_user.groups.keys():
        transactions = bonifici_by_user.get_group(user).sort_values("Timestamp").reset_index(drop=True)
        sequence_x_train, sequence_y_train = get_sequences_from_user(transactions, look_back)

        x.extend(sequence_x_train)
        y.extend(sequence_y_train)

        number_of_users += 1
        counter += 1
        # print("user:", counter, user)
    # print("Created", number_of_users, "users' sequences")
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

def create_train_set(look_back, scenario_type=constants.ALL_SCENARIOS):
    print("Preparing training set...")
    if constants.INJECTED_DATASET:
        path = "../datasets/train_63_users_" + scenario_type + "_scenario.csv"
        # path = "../datasets/train_167_users_" + scenario_type + "_scenario.csv"
        # path = "../datasets/train_529_users_" + scenario_type + "_scenario.csv"
        # path = "../datasets/train_696_users_" + scenario_type + "_scenario.csv"
    if constants.FRAUD_BUSTER_DATASET:
        path = "../datasets/fraud_buster_train_250_users_" + scenario_type + "_scenario.csv"
    if constants.REAL_DATASET:
        path = "../datasets/real_dataset_train_696_users.csv"

    print("Using as train: ", path)
    dataset_train = pd.read_csv(path, parse_dates=True)
    dataset_train = dataset_train.drop(["IP", "IBAN", "IBAN_CC", "CC_ASN"], axis=1)

    if constants.REAL_DATASET:
        users = constants.users_with_more_than_50_trx_train_5_trx_test
        dataset_train = dataset_train[dataset_train.UserID.isin(users)]

    x_train, y_train = create_sequences(dataset_train, look_back)
    return x_train, y_train

def create_test_set(look_back, scenario_type=constants.ALL_SCENARIOS):
    print("Preparing test set...")
    if constants.INJECTED_DATASET:
        path = "../datasets/test_63_users_" + scenario_type + "_scenario.csv"
        train_path = "../datasets/train_63_users_" + scenario_type + "_scenario.csv"
        # path = "../datasets/test_167_users_" + scenario_type + "_scenario.csv"
        # train_path = "../datasets/train_167_users_" + scenario_type + "_scenario.csv"
        # path = "../datasets/test_529_users_" + scenario_type + "_scenario.csv"
        # train_path = "../datasets/train_529_users_" + scenario_type + "_scenario.csv"
        # path = "../datasets/test_696_users_" + scenario_type + "_scenario.csv"
        # train_path = "../datasets/train_696_users_" + scenario_type + "_scenario.csv"
    if constants.FRAUD_BUSTER_DATASET:
        path = "../datasets/fraud_buster_test_250_users_" + scenario_type + "_scenario.csv"
    if constants.REAL_DATASET:
        path = "../datasets/real_dataset_test_696_users.csv"
        train_path = "../datasets/real_dataset_train_696_users.csv"

    print("Using as test: ", path)
    dataset_train = pd.read_csv(train_path, parse_dates=True)
    dataset_train = dataset_train.drop(["IP", "IBAN", "IBAN_CC", "CC_ASN"], axis=1)

    dataset_test = pd.read_csv(path, parse_dates=True)
    dataset_test = dataset_test.drop(["IP", "IBAN", "IBAN_CC", "CC_ASN"], axis=1)

    if constants.REAL_DATASET:
        users = constants.users_with_more_than_50_trx_train_5_trx_test
        dataset_train = dataset_train[dataset_train.UserID.isin(users)]
        dataset_test = dataset_test[dataset_test.UserID.isin(users)]

    print("len test set before injecting training transactions", len(dataset_test))
    # in test set, insert lookback transactions for each user taking them from training set.
    # in this way, all the transactions in test set will be used.
    dataset_test = insert_lookback_transactions_from_training_set(dataset_train, dataset_test, look_back)
    print("len test set after injecting training transactions", len(dataset_test))
    x_test, y_test = create_sequences(dataset_test, look_back)
    return x_test, y_test


if __name__ == "__main__":
    look_back = constants.LOOK_BACK
    x_train, y_train = create_train_set(look_back)
    x_test, y_test = create_test_set(look_back)

    np.save("../classification/dataset_" + str(look_back) + "_lookback/x_train.npy", x_train)
    np.save("../classification/dataset_" + str(look_back) + "_lookback/y_train.npy", y_train)
    np.save("../classification/dataset_" + str(look_back) + "_lookback/x_test.npy", x_test)
    np.save("../classification/dataset_" + str(look_back) + "_lookback/y_test.npy", y_test)
    print("Dataset saved.")