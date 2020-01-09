from models import LSTM_classifier, evaluation
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from dataset_creation import sequences_crafting_for_classification
from dataset_creation.constants import *
from sklearn.metrics import f1_score, recall_score, precision_score, balanced_accuracy_score, confusion_matrix
import random

def get_metrics(y_true, y_pred):
    c = confusion_matrix(y_true, y_pred)
    f = f1_score(y_true, y_pred)
    b = balanced_accuracy_score(y_true, y_pred)
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return c, f, b, p, r

def print_metrics(y_test, y_test_pred):
    c, f, b, p, r = get_metrics(y_test, y_test_pred)
    tn = c[0, 0]
    tp = c[1, 1]
    fp = c[0, 1]
    fn = c[1, 0]
    print("Confusion Matrix:")
    print(tn, fp)
    print(fn, tp)
    print("f1 score:", f)
    print("average accuracy:", b)
    print("precision:", p)
    print("recall (tpr):", r)
    print("fpr: ", fp / (fp + tn))

def get_predictions_for_each_model(lstm, rf, xg_reg, x, x_supervised):
    y_lstm = lstm.predict(x)
    y_rf = rf.predict_proba(x_supervised)
    y_xgb = xg_reg.predict_proba(x_supervised)
    return y_lstm, y_rf, y_xgb

# sum of the models outputs on validation set
# find the best threshold
# sum of the models outputs on test set
# adjust output based on threshold
def predict_test_based_on_sum(lstm, rf, xg_reg, x_val, x_val_supervised, y_val, x_test, x_test_supervised):
    y_val_pred_lstm, y_val_pred_rf, y_val_pred_xgb = get_predictions_for_each_model(lstm, rf, xg_reg, x_val, x_val_supervised)
    y_val_pred = y_val_pred_lstm.ravel() * ([0.333] * len(y_val_pred_lstm)) + \
                 y_val_pred_rf[:, 1].ravel() * ([0.333] * len(y_val_pred_rf)) + \
                 y_val_pred_xgb[:, 1].ravel() * ([0.333] * len(y_val_pred_xgb))

    threshold = evaluation.find_best_threshold_fixed_fpr(y_val, y_val_pred)

    y_pred_lstm, y_pred_rf, y_pred_xgb = get_predictions_for_each_model(lstm, rf, xg_reg, x_test, x_test_supervised)

    y_pred = y_pred_lstm.ravel() * ([0.333] * len(y_pred_lstm)) + \
             y_pred_rf[:, 1].ravel() * ([0.333] * len(y_pred_rf)) + \
             y_pred_xgb[:, 1].ravel() * ([0.333] * len(y_pred_xgb))

    y_test_pred = evaluation.adjusted_classes(y_pred, threshold)
    return y_test_pred

# each model gives his opinion and the output is based on the majority
def predict_test_based_on_voting(lstm, rf, xg_reg, x_val, x_val_supervised, y_val, x_test, x_test_supervised):
    y_val_pred_lstm, y_val_pred_rf, y_val_pred_xgb = get_predictions_for_each_model(lstm, rf, xg_reg, x_val,
                                                                                    x_val_supervised)

    # get threshold for each model
    threshold_lstm = evaluation.find_best_threshold_fixed_fpr(y_val, y_val_pred_lstm.ravel())
    threshold_rf = evaluation.find_best_threshold_fixed_fpr(y_val, y_val_pred_rf[:, 1])
    threshold_xgb = evaluation.find_best_threshold_fixed_fpr(y_val, y_val_pred_xgb[:, 1])

    # predicting test set
    y_pred_lstm, y_pred_rf, y_pred_xgb = get_predictions_for_each_model(lstm, rf, xg_reg, x_test, x_test_supervised)
    y_pred_lstm = y_pred_lstm.ravel()
    y_pred_rf = y_pred_rf[:, 1].ravel()
    y_pred_xgb = y_pred_xgb[:, 1].ravel()

    y_pred_lstm = evaluation.adjusted_classes(y_pred_lstm, threshold_lstm)
    y_pred_rf = evaluation.adjusted_classes(y_pred_rf, threshold_rf)
    y_pred_xgb = evaluation.adjusted_classes(y_pred_xgb, threshold_xgb)

    y_test_pred = []
    lstm_positive_votes = 0
    rf_positive_votes = 0
    xgb_positive_votes = 0

    for i in range(len(y_pred_lstm)):
        if y_pred_lstm[i] + y_pred_rf[i] + y_pred_xgb[i] >= 2:
            if y_pred_lstm[i] == 1:
                lstm_positive_votes += 1

            if y_pred_rf[i] == 1:
                rf_positive_votes += 1

            if y_pred_xgb[i] == 1:
                xgb_positive_votes += 1
            y_test_pred.append(1)
        else:
            y_test_pred.append(0)

    print("LSTM votes: ", lstm_positive_votes)
    print("RF votes: ", rf_positive_votes)
    print("XGboost votes: ", xgb_positive_votes)
    return y_test_pred

# for each model, get threshold and get mean of the output on validation set .
# on test set, calculate the distance from the calculated mean.
# The model that has the max distance will be the one who decides
def predict_test_based_on_more_confident(lstm, rf, xg_reg, x_val, x_val_supervised, y_val, x_test, x_test_supervised):
    y_val_pred_lstm, y_val_pred_rf, y_val_pred_xgb = get_predictions_for_each_model(lstm, rf, xg_reg, x_val,
                                                                                    x_val_supervised)

    # get threshold for each model
    threshold_lstm = evaluation.find_best_threshold_fixed_fpr(y_val, y_val_pred_lstm.ravel())
    threshold_rf = evaluation.find_best_threshold_fixed_fpr(y_val, y_val_pred_rf[:, 1])
    threshold_xgb = evaluation.find_best_threshold_fixed_fpr(y_val, y_val_pred_xgb[:, 1])

    # get mean for each model
    lstm_mean = np.array(y_val_pred_lstm.ravel()).mean()
    rf_mean = np.array(y_val_pred_rf[:, 1]).mean()
    xgb_mean = np.array(y_val_pred_xgb[:, 1]).mean()

    lstm_std = np.array(y_val_pred_lstm.ravel()).std()
    rf_std = np.array(y_val_pred_rf[:, 1]).std()
    xgb_std = np.array(y_val_pred_xgb[:, 1]).std()

    # predicting test set
    y_pred_lstm, y_pred_rf, y_pred_xgb = get_predictions_for_each_model(lstm, rf, xg_reg, x_test, x_test_supervised)
    y_pred_lstm = y_pred_lstm.ravel()
    y_pred_rf = y_pred_rf[:, 1].ravel()
    y_pred_xgb = y_pred_xgb[:, 1].ravel()

    y_test_pred = []
    num_decisions_taken_by_lstm = 0
    num_decisions_taken_by_rf = 0
    num_decisions_taken_by_xgb = 0
    for i in range(len(y_pred_lstm)):
        max_value = max(abs(y_pred_lstm[i] - lstm_mean) / lstm_std, abs(y_pred_rf[i] - rf_mean) / rf_std, abs(y_pred_xgb[i] - xgb_mean) / xgb_std)
        if abs(y_pred_lstm[i] - lstm_mean) / lstm_std == max_value:
            num_decisions_taken_by_lstm += 1
            y_test_pred.append(1 if y_pred_lstm[i] > threshold_lstm else 0)
        elif abs(y_pred_rf[i] - rf_mean) / rf_std == max_value:
            y_test_pred.append(1 if y_pred_rf[i] > threshold_rf else 0)
            num_decisions_taken_by_rf += 1
        else:
            num_decisions_taken_by_xgb += 1
            y_test_pred.append(1 if y_pred_xgb[i] > threshold_xgb else 0)

    print("Num decisions taken from lstm: ", num_decisions_taken_by_lstm)
    print("Num decisions taken by rf: ", num_decisions_taken_by_rf)
    print("Num decisions taken by xgb: ", num_decisions_taken_by_xgb)
    return y_test_pred


look_back = LOOK_BACK
print("Lookback using: ", look_back)
x_train, y_train = sequences_crafting_for_classification.create_train_set(look_back)

# train model for supervised models (xgboost/rf)
x_train_supervised = x_train[:, look_back, :]
y_train_supervised = y_train

print("Training models...")
lstm = LSTM_classifier.create_fit_model(x_train, y_train, look_back)

xg_reg = xgb.XGBClassifier()
xg_reg.fit(x_train_supervised, y_train_supervised)

rf = RandomForestClassifier()
rf.fit(x_train_supervised, y_train_supervised)

scenarios = [FIRST_SCENARIO, SECOND_SCENARIO, THIRD_SCENARIO, FOURTH_SCENARIO, FIFTH_SCENARIO, SIXTH_SCENARIO, SEVENTH_SCENARIO, EIGHTH_SCENARIO, NINTH_SCENARIO, ALL_SCENARIOS]
# scenarios = [FIRST_SCENARIO]
for scenario in scenarios:
    print("-------------------", scenario, "scenario --------------------------")
    x_test, y_test = sequences_crafting_for_classification.create_test_set(look_back, scenario)
    x_val, x_val_supervised, y_val, x_test, x_test_supervised, y_test = evaluation.get_val_test_set(x_test, y_test, look_back)

    '''
    print("----BASED ON SUM----")
    y_test_pred = predict_test_based_on_sum(lstm, rf, xg_reg, x_val, x_val_supervised, y_val, x_test, x_test_supervised)
    print_metrics(y_test, y_test_pred)

    print("----BASED ON MORE CONFIDENT----")
    y_test_pred = predict_test_based_on_more_confident(lstm, rf, xg_reg, x_val, x_val_supervised, y_val, x_test, x_test_supervised)
    print_metrics(y_test, y_test_pred)
    '''

    print("----BASED ON VOTING----")
    y_test_pred = predict_test_based_on_voting(lstm, rf, xg_reg, x_val, x_val_supervised, y_val, x_test, x_test_supervised)
    print_metrics(y_test, y_test_pred)