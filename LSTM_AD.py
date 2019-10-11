import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
# seaborn can generate several warnings, we ignore them
import warnings
from sklearn.model_selection import train_test_split
from models import LSTM
import config as cfg
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV
import wandb
from wandb.keras import WandbCallback
wandb.init(project="fraud-detection-thesis")

# in order to print all the columns
pd.set_option('display.max_columns', 100)
sns.set(style="white", color_codes=True)
sns.set_context(rc={"font.family":'sans',"font.size":24,"axes.titlesize":24,"axes.labelsize":24})
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------------------------

# ---------Main idea: creating an LSTM predictor for the next transaction of a user--------------
# The LSTM network is trained using sequences belonging to same user for all users

# -----------------------------------------------------------------------------------------------

IS_MODEL_SELECTION_ON = False


def get_series(transactions, look_back):
    x = []
    y = []
    while len(transactions.index) > look_back:
        x.append(transactions[0:look_back].values.tolist())
        y.append(transactions[look_back:look_back + 1].values[0]) # there is only one row
        transactions = transactions[look_back:]
    return x, y


# converting low level dataset to high level
def create_dataset(dataset, look_back=1):
    bonifici_by_user = dataset.groupby("UserID")
    x = []
    y = []

    # un utente è valido se ha più di lookback transazioni
    number_of_series = 0
    number_of_users = 0
    for user in bonifici_by_user.groups.keys():
        transactions = bonifici_by_user.get_group(user).sort_values("Timestamp")
        transactions = transactions.drop(["UserID", "Timestamp", "indice"], axis=1)
        series_x, series_y = get_series(transactions, look_back)
        x.extend(series_x)
        y.extend(series_y)
        number_of_users += 1
        number_of_series += len(series_y)

    print("Ci sono " + str(number_of_series) + " serie con " + str(number_of_users) + " utenti.")
    return np.asarray(x), np.asarray(y)


# reading the datasets
bonifici = pd.read_csv("datasets/bonifici_engineered.csv")


# --------------------------------- preparing training and test sets-----------------------------
print("Preparing training and test sets")

bonifici = bonifici.drop(["IDSessione", "IP", "IBAN", "IBAN_CC", "CC_ASN"], axis=1)

train, test = train_test_split(bonifici, test_size=0.1)
x_train, y_train = create_dataset(train, look_back=20)
x_test, y_test = create_dataset(test, look_back=20)

look_back = cfg.multi_step_lstm_config['look_back']
look_ahead = cfg.multi_step_lstm_config['look_ahead']
batch_size = cfg.multi_step_lstm_config['batch_size']
epochs = cfg.multi_step_lstm_config['n_epochs']
dropout = cfg.multi_step_lstm_config['dropout']
layers = cfg.multi_step_lstm_config['layers']
loss = cfg.multi_step_lstm_config['loss']
# optimizer = cfg.multi_step_lstm_config['optimizer']
shuffle = cfg.multi_step_lstm_config['shuffle']
patience = cfg.multi_step_lstm_config['patience']
validation = cfg.multi_step_lstm_config['validation']
learning_rate = cfg.multi_step_lstm_config['learning_rate']

vanilla_lstm = LSTM.VanillaLSTM(look_back=look_back, layers=layers, dropout=dropout, loss=loss, learning_rate=learning_rate)
model = vanilla_lstm.build_model()

print("Training model...")
history = LSTM.train_model(model, x_train, y_train, batch_size, epochs, shuffle, validation, ([], []), patience)
print("Training done.")