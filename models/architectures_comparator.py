from models import LSTM_classifier, evaluation
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from dataset_creation import sequences_crafting_for_classification
from dataset_creation.constants import *

look_back = LOOK_BACK
print("Using as lookback:", look_back)
x_train, y_train = sequences_crafting_for_classification.get_train_set()

# train model for supervised models (xgboost/rf)
x_train_supervised = x_train[:, look_back, :]
y_train_supervised = y_train

print("Training models...")
lstm = LSTM_classifier.create_fit_model(x_train, y_train, look_back)

'''
best_params_xgboost = {'subsample': 0.8, 'min_child_weight': 5, 'max_depth': 5, 'gamma': 1.5, 'colsample_bytree': 0.6}
xg_reg = xgb.XGBClassifier(subsample=best_params_xgboost["subsample"],
                           min_child_weight=best_params_xgboost["min_child_weight"],
                           max_depth=best_params_xgboost["max_depth"],
                           gamma=best_params_xgboost["gamma"],
                           colsample_bytree=best_params_xgboost["colsample_bytree"])
'''
xg_reg = xgb.XGBClassifier()
xg_reg.fit(x_train_supervised, y_train_supervised)

'''
best_params_rf = {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_features': 'auto', 'max_depth': 10, 'bootstrap': True}
rf = RandomForestClassifier(n_estimators=best_params_rf["n_estimators"],
                            min_samples_split=best_params_rf["min_samples_split"],
                            min_samples_leaf=best_params_rf["min_samples_leaf"],
                            max_features=best_params_rf["max_features"],
                            max_depth=best_params_rf["max_depth"],
                            bootstrap=best_params_rf["bootstrap"])
'''
rf = RandomForestClassifier()
rf.fit(x_train_supervised, y_train_supervised)

if DATASET_TYPE == INJECTED_DATASET:
    scenarios = [FIRST_SCENARIO, SECOND_SCENARIO, THIRD_SCENARIO, FOURTH_SCENARIO, FIFTH_SCENARIO, SIXTH_SCENARIO, SEVENTH_SCENARIO, EIGHTH_SCENARIO, NINTH_SCENARIO, ALL_SCENARIOS]
    # scenarios = [ALL_SCENARIOS]
    for scenario in scenarios:
        print("-------------------", scenario, "scenario --------------------------")
        x_test, y_test = sequences_crafting_for_classification.get_test_set(scenario=scenario)
        x_test_supervised = x_test[:, look_back, :]

        print("LSTM")
        y_pred_lstm = lstm.predict(x_test)
        evaluation.evaluate_n_times(y_test, y_pred_lstm)

        print("RF")
        y_pred_rf = rf.predict_proba(x_test_supervised)
        evaluation.evaluate_n_times(y_test, y_pred_rf[:, 1])

        print("Xgboost")
        y_pred_xgb = xg_reg.predict_proba(x_test_supervised)
        evaluation.evaluate_n_times(y_test, y_pred_xgb[:, 1])

        evaluation.print_jaccard_index(lstm, rf, xg_reg, x_test, y_test, look_back)


if DATASET_TYPE == REAL_DATASET:
    x_test, y_test = sequences_crafting_for_classification.get_test_set()
    x_test_supervised = x_test[:, look_back, :]

    print("LSTM")
    y_pred_lstm = lstm.predict(x_test)
    evaluation.evaluate_n_times(y_test, y_pred_lstm.ravel())

    print("RF")
    y_pred_rf = rf.predict_proba(x_test_supervised)
    evaluation.evaluate_n_times(y_test, y_pred_rf[:, 1])

    print("Xgboost")
    y_pred_xgb = xg_reg.predict_proba(x_test_supervised)
    evaluation.evaluate_n_times(y_test, y_pred_xgb[:, 1])

    evaluation.print_jaccard_index(lstm, rf, xg_reg, x_test, y_test, look_back)