import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from dataset_creation import sequences_crafting_for_classification, constants
from models import LSTM_classifier, evaluation


def model_selection(x_train, y_train):
    xg_reg = xgb.XGBClassifier()
    random_grid = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
    }

    xgb_random = RandomizedSearchCV(estimator=xg_reg,
                                    param_distributions=random_grid,
                                    n_iter=100,
                                    cv=3,
                                    verbose=2,
                                    random_state=42,
                                    n_jobs=-1)
    xgb_random.fit(x_train, y_train)

    print("Best params: ", xgb_random.best_params_)
    best_random = xgb_random.best_estimator_
    return best_random


def create_model(x_train, y_train):
    xg_reg = xgb.XGBClassifier()
    xg_reg.fit(x_train, y_train)
    return xg_reg


look_back = constants.LOOK_BACK
'''
# dataset_train = pd.read_csv("../datasets/real_dataset_train_58561_users.csv", parse_dates=True)
dataset_train = pd.read_csv("../datasets/train_529_users_ALL_scenario.csv", parse_dates=True)
dataset_train = dataset_train.drop(["IP", "IBAN", "IBAN_CC", "CC_ASN"], axis=1)
dataset_train = dataset_train.drop(["Timestamp", "UserID"], axis=1)
y_train = dataset_train.isFraud
x_train_sup = dataset_train.drop(["isFraud", "mean_amount_30_window", "mean_amount_7_window", "stdev_amount_7_window", "stdev_amount_30_window", "mean_amount_1000_window", "stdev_amount_1000_window"], axis=1)

# dataset_test = pd.read_csv("../datasets/real_dataset_test_58561_users.csv", parse_dates=True)
dataset_test = pd.read_csv("../datasets/test_529_users_ALL_scenario.csv", parse_dates=True)
dataset_test = dataset_test.drop(["IP", "IBAN", "IBAN_CC", "CC_ASN"], axis=1)
y_test = dataset_test.isFraud
dataset_test = dataset_test.drop(["Timestamp", "UserID"], axis=1)
x_test_sup = dataset_test.drop(["isFraud", "mean_amount_30_window", "mean_amount_7_window", "stdev_amount_7_window", "stdev_amount_30_window", "mean_amount_1000_window", "stdev_amount_1000_window"], axis=1)
'''
'''
dataset_train = pd.read_csv("../datasets/train_63_users_ALL_scenario.csv", parse_dates=True)
dataset_train = dataset_train.drop(["IP", "IBAN", "IBAN_CC", "CC_ASN"], axis=1)
dataset_test = pd.read_csv("../datasets/test_63_users_ALL_scenario.csv", parse_dates=True)
dataset_test = dataset_test.drop(["IP", "IBAN", "IBAN_CC", "CC_ASN"], axis=1)

prova_test = dataset_test.sort_values("Timestamp").reset_index(drop=True)
prova_train = dataset_train.sort_values("Timestamp").reset_index(drop=True)
sequence_x_train, sequence_y_train = sequences_crafting_for_classification.create_sequences(prova_train, 1)
sequence_x_test, sequence_y_test = sequences_crafting_for_classification.create_sequences(prova_test, 1)

x_train_sup = sequence_x_train[:, 1, :]
x_test_sup = sequence_x_test[:, 1, :]

print("Fitting model...")
model = create_model(x_train_sup, sequence_y_train)
print("Evaluating model...")
y_pred = model.predict_proba(x_test_sup)
evaluation.evaluate(sequence_y_test, y_pred[:, 1])
'''

x_train, y_train = sequences_crafting_for_classification.create_train_set(look_back)
x_test, y_test = sequences_crafting_for_classification.create_test_set(look_back)

# adapt train and test set to supervised learning without time windows
x_train_sup = x_train[:, look_back, :]
x_test_sup = x_test[:, look_back, :]


print("Fitting model...")
# model = model_selection(x_train, y_train)
model = create_model(x_train_sup, y_train)

print("Evaluating model...")
y_pred = model.predict_proba(x_test_sup)
evaluation.evaluate(y_test, y_pred[:, 1])