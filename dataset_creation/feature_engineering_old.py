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
# --------- Feature engineering module for the unlabeled dataset --------------------------------
# -----------------------------------------------------------------------------------------------

# reading the datasets
bonifici = pd.read_csv("/home/mpapale/thesis/datasets/old_bonifici.csv", parse_dates=True, sep=";")
bonifici.set_index('index',inplace=True)

# dropping columns with useless data
useless_features = ["CAP", "Status", "Paese", "Provincia", "Nazione", "IDTransazione", "CRO", "Causale", "Valuta", "ProfSicurezza", "NumConto", "ABI", "CAB", "Intestatario", "Indirizzo"]
bonifici = bonifici.drop(useless_features, axis=1)
# in future, try to use these features
bonifici = bonifici.drop(["DataValuta", "DataEsecuzione", "Nominativo", "TipoOperazione"], axis=1)

# bonifici.Timestamp = pd.to_datetime(bonifici.Timestamp)
bonifici.NumConfermaSMS = bonifici.NumConfermaSMS.eq('Si').astype(int)

print("Rescaling importo...")
# rescaling "Importo"
x = bonifici[["Importo"]].to_numpy()
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
bonifici[["Importo"]] = x_scaled

print("Changing MsgErrore to boolean...")
# MsgErrore changed to boolean
bonifici.MsgErrore.fillna(0, inplace=True)
bonifici["MsgErrore"] = bonifici["MsgErrore"].apply(lambda x : 1 if x != 0 else 0)

print("Creating isItalianSender and isItalianReceiver...")
# creating "isItalianSender", "isItalianReceiver"
bonifici["isItalianSender"] = np.ones(len(bonifici.index))
bonifici["isItalianReceiver"] = np.ones(len(bonifici.index))
i = 0
for index, row in bonifici[["CC_ASN", "IBAN_CC"]].iterrows():
    print(i)
    i += 1
    if row["CC_ASN"][:2] != "IT":
        bonifici.at[index, "isItalianSender"] = 0
    if row["IBAN_CC"] != "IT":
        bonifici.at[index, "isItalianReceiver"] = 0
bonifici["isItalianSender"] = pd.to_numeric(bonifici["isItalianSender"], downcast='integer')
bonifici["isItalianReceiver"] = pd.to_numeric(bonifici["isItalianReceiver"], downcast='integer')

# todo: creare timestamp delta di quanto tempo è passato dall'ultima tranasazione
print("Creating count_trx_iban, is_new_asn_cc, is_new_iban, is_new_iban_cc, is_new_ip...")
# creating count_trx_iban, is_new_asn_cc, is_new_iban, is_new_iban_cc
bonifici_by_user = bonifici.groupby("UserID")

bonifici["count_trx_iban"] = np.zeros(len(bonifici.index))
bonifici["is_new_asn_cc"] = np.zeros(len(bonifici.index))
bonifici["is_new_iban"] = np.zeros(len(bonifici.index))
bonifici["is_new_iban_cc"] = np.zeros(len(bonifici.index))
bonifici["is_new_ip"] = np.zeros(len(bonifici.index))
# for each transaction of the user:
    #   - count the number of preceding transactions to the same iban
    #   - check if the asn_cc is new
    #   - check if the iban receiver is new
    #   - check if the iban_cc is new
    #   - check if the ip of the user is new
counter = 0
for user in bonifici_by_user.groups.keys():
    print(counter)
    counter += 1
    # order by time
    group = bonifici_by_user.get_group(user).sort_values(by='Timestamp', ascending=True)

    for i in range(len(group)):
        count_trx_iban = 0
        is_new_asn_cc = 1
        is_new_iban = 1
        is_new_iban_cc = 1
        is_new_ip = 1
        for j in range(0, i):
            if group.iloc[i]["IBAN"] == group.iloc[j]["IBAN"]:
                count_trx_iban += 1
            if group.iloc[i]["CC_ASN"] == group.iloc[j]["CC_ASN"]:
                is_new_asn_cc = 0
            if group.iloc[i]["IBAN"] == group.iloc[j]["IBAN"]:
                is_new_iban = 0
            if group.iloc[i]["IBAN_CC"] == group.iloc[j]["IBAN_CC"]:
                is_new_iban = 0
            if group.iloc[i]["IP"] == group.iloc[j]["IP"]:
                is_new_ip = 0
        bonifici.at[group.iloc[i:i + 1].index[0], "count_trx_iban"] = count_trx_iban
        bonifici.at[group.iloc[i:i + 1].index[0], "is_new_asn_cc"] = is_new_asn_cc
        bonifici.at[group.iloc[i:i + 1].index[0], "is_new_iban"] = is_new_iban
        bonifici.at[group.iloc[i:i + 1].index[0], "is_new_iban_cc"] = is_new_iban_cc
        bonifici.at[group.iloc[i:i + 1].index[0], "is_new_ip"] = is_new_ip

bonifici["count_trx_iban"] = pd.to_numeric(bonifici["count_trx_iban"], downcast='integer')
bonifici["is_new_asn_cc"] = pd.to_numeric(bonifici["is_new_asn_cc"], downcast='integer')
bonifici["is_new_iban"] = pd.to_numeric(bonifici["is_new_iban"], downcast='integer')
bonifici["is_new_iban_cc"] = pd.to_numeric(bonifici["is_new_iban_cc"], downcast='integer')
bonifici["is_new_ip"] = pd.to_numeric(bonifici["is_new_ip"], downcast='integer')

# saving the dataset
bonifici.to_csv("/home/mpapale/thesis/datasets/old_bonifici_engineered.csv")