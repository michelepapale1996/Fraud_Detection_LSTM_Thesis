import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
sns.set(style="white", color_codes=True)
sns.set_context(rc={"font.family":'sans',"font.size":24,"axes.titlesize":24,"axes.labelsize":24})
import matplotlib.pyplot as plt
# seaborn can generate several warnings, we ignore them
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV
# in order to print all the columns
pd.set_option('display.max_columns', 100)

IS_MODEL_SELECTION_ON = False

# reading the datasets
bonifici = pd.read_csv("datasets/quiubi_bonifici.csv")
segnalaz = pd.read_csv("datasets/bonifici_segnalaz.csv")

# dropping columns with useless data
bonifici = bonifici.drop(["DataValuta", "DataEsecuzione", "CAP", "Servizio", "Status", "Paese", "Provincia", "Nazione", "IDTransazione", "CRO", "Causale", "Valuta", "ProfSicurezza", "NumConto", "ABI", "CAB", "Intestatario", "Indirizzo"], axis=1)
segnalaz = segnalaz.drop(["DataValuta", "DataEsecuzione", "CAP", "Servizio", "Status", "Paese", "Provincia", "Nazione", "IDTransazione", "CRO", "Causale", "Valuta", "ProfSicurezza", "NumConto", "ABI", "CAB", "Intestatario", "Indirizzo"], axis=1)
bonifici = bonifici.drop(["MsgErrore", "Nominativo", "TipoOperazione"], axis=1)
segnalaz = segnalaz.drop(["MsgErrore", "Nominativo", "TipoOperazione"], axis=1)

# ci sono due transazioni con lo stesso indice -> ne modifico una
bonifici.at[364155, "indice"] = "00000000000000000000000000"

bonifici.set_index('indice',inplace=True)
segnalaz.set_index('indice',inplace=True)

# datasets merge into bonifici and adding isFraud feature
bonifici.loc[:, "isFraud"] = np.zeros(len(bonifici.index))
for index, row in segnalaz.iterrows():
    if index not in bonifici.index:
        # add row to dataframe
        bonifici.loc[index] = row
    # set isFraud to true
    bonifici.loc[index, "isFraud"] = 1
bonifici["isFraud"] = pd.to_numeric(bonifici["isFraud"], downcast='integer')
print("Solo il " + str((len(bonifici[bonifici["isFraud"] == 1].index) / len(bonifici.index)) * 100) + "% delle transazioni sono frodi.")

