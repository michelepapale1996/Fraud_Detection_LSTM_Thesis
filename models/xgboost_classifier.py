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

# x_train, y_train = sequences_crafting_for_classification.create_train_set(look_back)
# x_test, y_test = sequences_crafting_for_classification.create_test_set(look_back)

x_train = np.load("/home/mpapale/thesis/classification/dataset_" + str(look_back) + "_lookback/x_train_50_trx_per_user.npy")
y_train = np.load("/home/mpapale/thesis/classification/dataset_" + str(look_back) + "_lookback/y_train_50_trx_per_user.npy")
x_test = np.load("/home/mpapale/thesis/classification/dataset_" + str(look_back) + "_lookback/x_test_50_trx_per_user.npy")
y_test = np.load("/home/mpapale/thesis/classification/dataset_" + str(look_back) + "_lookback/y_test_50_trx_per_user.npy")

# adapt train and test set to supervised learning without time windows
x_train_sup = x_train[:, look_back, :]
x_test_sup = x_test[:, look_back, :]


print("Fitting model...")
# model = model_selection(x_train_sup, y_train)
model = create_model(x_train_sup, y_train)

print("Evaluating model...")
y_pred = model.predict_proba(x_test_sup)
evaluation.evaluate_n_times(y_test, y_pred[:, 1])