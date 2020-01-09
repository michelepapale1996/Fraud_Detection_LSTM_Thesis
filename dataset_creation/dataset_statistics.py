# dataframe management
import pandas as pd
# numerical computation
import numpy as np
# visualization library
import seaborn as sns
sns.set(style="white", color_codes=True)
sns.set_context(rc={"font.family":'sans',"font.size":24,"axes.titlesize":24,"axes.labelsize":24})
# import matplotlib and allow it to plot inline
import matplotlib.pyplot as plt

# seaborn can generate several warnings, we ignore them
import warnings
warnings.filterwarnings("ignore")
#in order to prin all the columns
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 10000)
from datetime import timedelta, date


#reading the datasets
bonifici = pd.read_csv("../datasets/quiubi_bonifici.csv")
segnalaz = pd.read_csv("../datasets/bonifici_segnalaz.csv")
bonifici = bonifici.drop(["CAP", "Servizio", "Status", "Paese", "Provincia", "Nazione", "IDTransazione", "CRO", "Causale", "Valuta", "ProfSicurezza", "NumConto", "ABI", "CAB", "Intestatario", "Indirizzo"], axis=1)
bonifici.set_index('indice',inplace=True)
segnalaz.set_index('indice',inplace=True)
bonifici = bonifici.drop(["DataValuta", "DataEsecuzione", "Nominativo", "TipoOperazione"], axis=1)
segnalaz = segnalaz.drop(["DataValuta", "DataEsecuzione", "Nominativo", "TipoOperazione"], axis=1)
bonifici.Timestamp = pd.to_datetime(bonifici.Timestamp)

# datasets merge into bonifici
bonifici.loc[:, "isFraud"] = np.zeros(len(bonifici.index))
for index, row in segnalaz.iterrows():
    if index in bonifici.index:
        bonifici.loc[index, "isFraud"] = 1
    else:
        bonifici.append(row)
bonifici.loc[:, "isFraud"] = pd.to_numeric(bonifici["isFraud"], downcast='integer')

bonifici_by_user = bonifici.groupby("UserID")

users = []
thirty_days = timedelta(30)
first_date = min(bonifici.Timestamp)
last_date = max(bonifici.Timestamp)
last_date_train_set = last_date - thirty_days

# find the users with at least x transactions in training and at least y transactions in test set
for user in bonifici_by_user.groups.keys():
    group = bonifici_by_user.get_group(user).sort_values(by='Timestamp', ascending=True)
    group_train = group[group.Timestamp < last_date_train_set]
    group_test = group[group.Timestamp >= last_date_train_set]

    if len(group_test) > 5 and len(group_train) > 500:
        print("user: ", user, ", len: ", len(group), "len test: ", len(group_test))
        users.append(user)
print(users)