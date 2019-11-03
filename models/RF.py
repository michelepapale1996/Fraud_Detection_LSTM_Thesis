import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
sns.set(style="white", color_codes=True)
sns.set_context(rc={"font.family":'sans',"font.size":24,"axes.titlesize":24,"axes.labelsize":24})
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

print("Preparing training and test sets...")
bonifici = pd.read_csv("../datasets/bonifici_engineered.csv", parse_dates=True)
bonifici.set_index('indice', inplace=True)
bonifici = bonifici.drop(["IDSessione", "IP", "IBAN", "IBAN_CC", "CC_ASN"], axis=1)

# --------------------------creating train and test sets-----------------------------------------
# Labels are the values we want to predict
labels = np.array(bonifici['isFraud'])
# Remove the labels from the features

# Saving feature names for later use
feature_list = ['Importo',
        'NumConfermaSMS',
        'isItalianSender',
        'isItalianReceiver',
        'count_trx_iban',
        'is_new_asn_cc',
        'is_new_iban',
        'is_new_iban_cc',
        'is_new_ip']

bonifici = bonifici[feature_list]

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