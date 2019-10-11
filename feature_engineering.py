import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
# seaborn can generate several warnings, we ignore them
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split

# in order to print all the columns
pd.set_option('display.max_columns', 100)
sns.set(style="white", color_codes=True)
sns.set_context(rc={"font.family":'sans',"font.size":24,"axes.titlesize":24,"axes.labelsize":24})


# -----------------------------------------------------------------------------------------------
# -------------------------- Feature engineering module -----------------------------------------
# -----------------------------------------------------------------------------------------------

# reading the datasets
bonifici = pd.read_csv("datasets/quiubi_bonifici.csv")
segnalaz = pd.read_csv("datasets/bonifici_segnalaz.csv")
bonifici.set_index('indice',inplace=True)
segnalaz.set_index('indice',inplace=True)

# dropping columns with useless data
useless_features = ["CAP", "Servizio", "Status", "Paese", "Provincia", "Nazione", "IDTransazione", "CRO", "Causale", "Valuta", "ProfSicurezza", "NumConto", "ABI", "CAB", "Intestatario", "Indirizzo"]
bonifici = bonifici.drop(useless_features, axis=1)
segnalaz = segnalaz.drop(useless_features, axis=1)
bonifici = bonifici.drop(["DataValuta", "DataEsecuzione", "MsgErrore", "Nominativo", "TipoOperazione"], axis=1)
segnalaz = segnalaz.drop(["DataValuta", "DataEsecuzione", "MsgErrore", "Nominativo", "TipoOperazione"], axis=1)

# datasets merge into bonifici
bonifici["isFraud"] = np.zeros(len(bonifici.index))
for index, row in segnalaz.iterrows():
    if index in bonifici.index:
        bonifici.loc[index, "isFraud"] = 1
    else:
        # print(index)
        bonifici.append(row)
bonifici["isFraud"] = pd.to_numeric(bonifici["isFraud"], downcast='integer')

bonifici.Timestamp = pd.to_datetime(bonifici.Timestamp)
bonifici.NumConfermaSMS = bonifici.NumConfermaSMS.eq('Si').astype(int)

# rescaling "Importo"
x = bonifici[["Importo"]].to_numpy()
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
bonifici[["Importo"]] = x_scaled

# creating "isItalianSender" and "isItalianReceiver"
bonifici.loc[:, "isItalianSender"] = np.ones(len(bonifici.index))
bonifici.loc[:, "isItalianReceiver"] = np.ones(len(bonifici.index))
for index, row in bonifici[["CC_ASN", "IBAN_CC"]].iterrows():
    if row["CC_ASN"][:2] != "IT":
        bonifici.at[index, "isItalianSender"] = 0
    if row["IBAN_CC"] != "IT":
        bonifici.at[index, "isItalianReceiver"] = 0
bonifici["isItalianSender"] = pd.to_numeric(bonifici["isItalianSender"], downcast='integer')
bonifici["isItalianReceiver"] = pd.to_numeric(bonifici["isItalianReceiver"], downcast='integer')

# saving the dataset
bonifici.to_csv("datasets/bonifici_engineered.csv")