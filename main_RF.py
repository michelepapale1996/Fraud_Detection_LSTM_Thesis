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

# ------------------------------ Feature engineering ------------------------------
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

'''
# --------------------------creating train and test sets-----------------------------------------

# Labels are the values we want to predict
labels = np.array(bonifici['isFraud'])
# Remove the labels from the features
bonifici = bonifici[["Importo", "NumConfermaSMS", "isItalianSender", "isItalianReceiver"]]
# Saving feature names for later use
feature_list = ["Importo", "NumConfermaSMS", "isItalianSender", "isItalianReceiver"]
# Convert to numpy array
features = np.array(bonifici)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

print("train_labels")
print(np.unique(train_labels, return_counts = True))
print("test_labels")
print(np.unique(test_labels, return_counts = True))

if IS_MODEL_SELECTION_ON:
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=2, stop=20, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    print("Starting CV...")
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)
    # Fit the random search model
    rf_random.fit(train_features, train_labels)

    print("Best params found: ")
    print(rf_random.best_params_)

    bestParams = rf_random.best_params_
else:
    bestParams = {'n_estimators': 16, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 10, 'bootstrap': True}

rf = RandomForestClassifier(
    n_estimators = bestParams["n_estimators"],
    max_features = bestParams["max_features"],
    max_depth = bestParams["max_depth"],
    min_samples_split = bestParams["min_samples_split"],
    min_samples_leaf = bestParams["min_samples_leaf"],
    bootstrap = bestParams["bootstrap"],
    random_state = 42)

# Train the model on training data
rf.fit(train_features, train_labels)
predicted_labels = rf.predict(test_features)

# ---------------performance indicators------------------ 
f1 = f1_score(test_labels, predicted_labels)
auc = roc_auc_score(test_labels, predicted_labels)
precision, recall, thresholds = precision_recall_curve(test_labels, predicted_labels)
fpr, tpr, thresholds = roc_curve(test_labels, predicted_labels)
print('f1_score: %.3f' % f1)
print('AUC: %.3f' % auc)


def print_roc_curve():
    # plot no skill
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')
    plt.title("Roc Curve")
    # show the plot
    plt.show()


def print_precision_recall_curve():
    # plot no skill
    plt.plot([0, 1], [0.5, 0.5], linestyle='--')
    # plot the precision-recall curve for the model
    plt.plot(recall, precision, marker='.')
    plt.title("Precision Recall Curve")
    # show the plot
    plt.show()


print_roc_curve()
print_precision_recall_curve()

'''