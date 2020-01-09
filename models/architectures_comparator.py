from models import LSTM_classifier, evaluation
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import pandas as pd
from dataset_creation import sequences_crafting_for_classification
from dataset_creation.constants import *

def print_jouden_statistics(lstm, rf, xg_reg, x_test, y_test, look_back):
    x_val, x_val_supervised, y_val, x_test, x_test_supervised, y_test = evaluation.get_val_test_set(x_test, y_test, look_back)

    y_pred_val = lstm.predict(x_val)
    y_pred_test = lstm.predict(x_test)
    lstm_indices_found = get_found_frauds_indices(y_pred_val, y_pred_test, y_val, y_test)

    y_pred_val = rf.predict_proba(x_val_supervised)
    y_pred_test = rf.predict_proba(x_test_supervised)
    rf_indices_found = get_found_frauds_indices(y_pred_val[:, 1], y_pred_test[:, 1], y_val, y_test)

    y_pred_val = xg_reg.predict_proba(x_val_supervised)
    y_pred_test = xg_reg.predict_proba(x_test_supervised)
    xgboost_indices_found = get_found_frauds_indices(y_pred_val[:, 1], y_pred_test[:, 1], y_val, y_test)

    get_jouden_statistic(lstm_indices_found, rf_indices_found, xgboost_indices_found)

def get_found_frauds_indices(y_val_pred, y_test_pred, y_val, y_test):
    threshold = evaluation.find_best_threshold_fixed_fpr(y_val, y_val_pred)

    y_pred_test = np.array(evaluation.adjusted_classes(y_test_pred, threshold))
    indices = set(np.where(y_pred_test == 1)[0])
    real_indices = set(np.where(y_test == 1)[0])
    indices_found = indices.intersection(real_indices)
    return indices_found

def get_jouden_statistic(lstm_indices, rf_indices, xgboost_indices):
    print("--Differences on the frauds found")
    print("---------------------------------")
    print("xgboost founds ", len(xgboost_indices), " frauds")
    print("rf founds ", len(rf_indices), " frauds")
    print("lstm founds ", len(lstm_indices), " frauds")
    print("---------------------------------")
    print("xgboost - rf")
    print("Intersection: ", len(xgboost_indices.intersection(rf_indices)), "Union: ", len(xgboost_indices.union(rf_indices)))
    print("Jouden: ", len(xgboost_indices.intersection(rf_indices))/len(xgboost_indices.union(rf_indices)))

    print("xgboost - lstm")
    print("Frauds found by lstm and not by xgboost:", len(lstm_indices - xgboost_indices))
    print("Intersection: ", len(xgboost_indices.intersection(lstm_indices)), "Union: ", len(xgboost_indices.union(lstm_indices)))
    print("Jouden: ", len(xgboost_indices.intersection(lstm_indices)) / len(xgboost_indices.union(lstm_indices)))

    print("rf - lstm")
    print("Frauds found by lstm and not by rf:", len(lstm_indices - rf_indices))
    print("Intersection: ", len(rf_indices.intersection(lstm_indices)), "Union: ", len(rf_indices.union(lstm_indices)))
    print("Jouden: ", len(rf_indices.intersection(lstm_indices)) / len(rf_indices.union(lstm_indices)))


look_back = LOOK_BACK
print("Using as lookback:", look_back)
x_train, y_train = sequences_crafting_for_classification.create_train_set(look_back)
# train model for supervised models (xgboost/rf)
x_train_supervised = x_train[:, look_back, :]
y_train_supervised = y_train

print("Training models...")
lstm = LSTM_classifier.create_fit_model(x_train, y_train, look_back)

best_params_xgboost = {'subsample': 0.6, 'min_child_weight': 10, 'max_depth': 3, 'gamma': 1.5, 'colsample_bytree': 0.8}

xg_reg = xgb.XGBClassifier(subsample=best_params_xgboost["subsample"],
                           min_child_weight=best_params_xgboost["min_child_weight"],
                           max_depth=best_params_xgboost["max_depth"],
                           gamma=best_params_xgboost["gamma"],
                           colsample_bytree=best_params_xgboost["colsample_bytree"])
xg_reg.fit(x_train_supervised, y_train_supervised)

best_params_rf = {'n_estimators': 2000, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 10, 'bootstrap': True}
rf = RandomForestClassifier(n_estimators=best_params_rf["n_estimators"],
                            min_samples_split=best_params_rf["min_samples_split"],
                            min_samples_leaf=best_params_rf["min_samples_leaf"],
                            max_features=best_params_rf["max_features"],
                            max_depth=best_params_rf["max_depth"],
                            bootstrap=best_params_rf["bootstrap"])
rf.fit(x_train_supervised, y_train_supervised)

if INJECTED_DATASET:
    scenarios = [FIRST_SCENARIO, SECOND_SCENARIO, THIRD_SCENARIO, FOURTH_SCENARIO, FIFTH_SCENARIO, SIXTH_SCENARIO, SEVENTH_SCENARIO, EIGHTH_SCENARIO, NINTH_SCENARIO, ALL_SCENARIOS]
    scenarios = [ALL_SCENARIOS]
    for scenario in scenarios:
        print("-------------------", scenario, "scenario --------------------------")
        x_test, y_test = sequences_crafting_for_classification.create_test_set(look_back, scenario)
        x_test_supervised = x_test[:, look_back, :]

        print("LSTM")
        y_pred_lstm = lstm.predict(x_test)
        evaluation.evaluate(y_test, y_pred_lstm)

        print("RF")
        y_pred_rf = rf.predict_proba(x_test_supervised)
        evaluation.evaluate(y_test, y_pred_rf[:, 1])

        print("Xgboost")
        y_pred_xgb = xg_reg.predict_proba(x_test_supervised)
        evaluation.evaluate(y_test, y_pred_xgb[:, 1])

        # Finding Jouden statistics
        print_jouden_statistics(lstm, rf, xg_reg, x_test, y_test, look_back)


if REAL_DATASET:
    x_test, y_test = sequences_crafting_for_classification.create_test_set(look_back)
    x_test_supervised = x_test[:, look_back, :]

    print("LSTM")
    y_pred_lstm = lstm.predict(x_test)
    evaluation.evaluate(y_test, y_pred_lstm)

    print("RF")
    y_pred_rf = rf.predict_proba(x_test_supervised)
    evaluation.evaluate(y_test, y_pred_rf[:, 1])

    print("Xgboost")
    y_pred_xgb = xg_reg.predict_proba(x_test_supervised)
    evaluation.evaluate(y_test, y_pred_xgb[:, 1])

    # Finding Jouden statistics
    print_jouden_statistics(lstm, rf, xg_reg, x_test, y_test, look_back)