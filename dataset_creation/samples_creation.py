import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
# seaborn can generate several warnings, we ignore them
import warnings

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
# returns numpy array
def get_sequences(transactions, look_back):
    cols = ['Importo', 'NumConfermaSMS', 'isItalianSender', 'isItalianReceiver']
    genuine_transactions = transactions.loc[transactions.isFraud == 0, cols]
    sequences = []
    labels = []

    pointer = 0
    while pointer < len(genuine_transactions.index) - look_back:
        sequence = genuine_transactions[pointer: pointer + look_back]
        index_last_genuine_transaction = genuine_transactions.index[pointer + look_back - 1]

        # get the next transaction
        i = transactions.index.get_loc(index_last_genuine_transaction)
        sequence = sequence.append(transactions.iloc[i + 1][cols])
        labels.append(transactions.iloc[i + 1]["isFraud"])
        sequences.append(sequence.values)
        pointer += 1
    return sequences, labels


# converting low level dataset (transactions of all users)
# to high level (sequences of transactions with at least length = look_back + 1)
def create_dataset(dataset, look_back=1):
    print("Creating samples using as look_back = " + str(look_back))
    bonifici_by_user = dataset.groupby("UserID")
    x = []
    y = []

    # un utente è valido se ha più di lookback transazioni
    number_of_sequences = 0
    number_of_users = 0
    for user in bonifici_by_user.groups.keys():
        transactions = bonifici_by_user.get_group(user).sort_values("Timestamp")
        transactions = transactions.drop(["UserID", "Timestamp"], axis=1)
        sequence_x, sequence_y = get_sequences(transactions, look_back)
        x.extend(sequence_x)
        y.extend(sequence_y)

        number_of_users += 1
        number_of_sequences += len(sequence_y)

    print("There are " + str(number_of_sequences) + " sequences with " + str(number_of_users) + " users.")
    return np.asarray(x), np.asarray(y)


def create_dataset_for_single_user(dataset, look_back=1):
    print("Creating samples using as look_back = " + str(look_back))
    # take the transactions of the user with most transactions
    transactions = dataset.loc[dataset["UserID"] == "e0e56529cd38cb40f414bbecfde594d5"].sort_values("Timestamp")

    transactions = transactions.drop(["UserID", "Timestamp"], axis=1)
    sequence_x, sequence_y = get_sequences(transactions, look_back)

    # return np.asarray(sequence_x), np.asarray(sequence_y)
    return sequence_x, sequence_y


print("Preparing training and test sets...")
bonifici = pd.read_csv("../datasets/bonifici_engineered.csv", parse_dates=True)
bonifici.set_index('indice', inplace=True)
bonifici = bonifici.drop(["IDSessione", "IP", "IBAN", "IBAN_CC", "CC_ASN"], axis=1)
x, y = create_dataset(bonifici, 25)