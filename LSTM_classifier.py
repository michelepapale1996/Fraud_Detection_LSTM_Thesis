import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
# seaborn can generate several warnings, we ignore them
import warnings
from sklearn.model_selection import TimeSeriesSplit
from metrics import f1_m, precision_m, recall_m
from sklearn.metrics import f1_score
from models import LSTM

import config as cfg
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
import wandb
from wandb.keras import WandbCallback

wandb.init(project="fraud-detection-thesis")

# in order to print all the columns
pd.set_option('display.max_columns', 100)
sns.set(style="white", color_codes=True)
sns.set_context(rc={"font.family": 'sans', "font.size": 24, "axes.titlesize": 24, "axes.labelsize": 24})
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------------------------
# --------------------------Main idea: creating an LSTM classifier ------------------------------
# -----------------------------------------------------------------------------------------------

# experiments
repeats = 2
epochs = [3]
look_backs = [900]
batch_size = [1, 5]
layers = [{'input': 10, 'hidden1': 10, 'hidden2': 10, 'output': 1}]


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
        print(number_of_users)
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


# run a repeated experiment with a fixed:
# - the experiment is repeated repeats times
# - number of epochs
# - sequence length = look_back
def experiment(dataset, repeats, epochs, look_back):
    # x, y = create_dataset(dataset, look_back)
    x, y = create_dataset(dataset, look_back)
    report = pd.DataFrame([], columns=['Loss', 'Accuracy', 'F1_score', 'Precision', 'Recall'])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    print("y train test...")
    print(y_train, y_test)

    for r in range(repeats):
        print("--------------------------------------------------")
        print("Current iteration: " + str(r))
        print("Current number of epochs: " + str(epochs))
        print("Current number of look_backs: " + str(look_back))
        print("Current train size: " + str(x_train.shape))
        print("Current test size: " + str(x_test.shape))

        lstm_classifier = LSTM.LstmClassifier(
            look_back=look_back,
            layers=layers[0],
            dropout=0,
            loss='binary_crossentropy',
            learning_rate=0.01,
            num_features=x_train.shape[2],
            metrics=['acc', f1_m, precision_m, recall_m])
        model = lstm_classifier.build_model()

        LSTM.train_model(model=model, x_train=x_train, y_train=y_train, batch_size=batch_size, epochs=epochs, shuffle=False)
        loss, accuracy, f1, precision, recall = model.evaluate(x_test, y_test, verbose=0)

        yhat = model.predict(x_test)
        print(yhat)
        print(loss, accuracy, f1, precision, recall)
        report = report.append({'Loss': loss, 'Accuracy': accuracy, 'F1_score': f1, 'Precision': precision, 'Recall': recall}, ignore_index=True)

    return report, x, y


print("Preparing training and test sets...")
# reading the datasets
bonifici = pd.read_csv("datasets/bonifici_engineered.csv", parse_dates=True)
bonifici.set_index('indice', inplace=True)
bonifici = bonifici.drop(["IDSessione", "IP", "IBAN", "IBAN_CC", "CC_ASN"], axis=1)

results = pd.DataFrame([], columns=['Loss', 'Accuracy', 'F1_score', 'Precision', 'Recall'])
for e in epochs:
    for l in look_backs:
        result, x, y = experiment(bonifici, repeats, e, l)
        results.append(result)

# summarize results
print(results.describe())