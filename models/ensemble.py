from models import LSTM_classifier, evaluation
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from dataset_creation import sequences_crafting_for_classification
from dataset_creation.constants import *
import random

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
def predict_test_based_on_voting(lstm, rf, xg_reg, x_val, x_val_supervised, y_val, x_test, x_test_supervised, y_test):
    y_val_pred_lstm, y_val_pred_rf, y_val_pred_xgb = get_predictions_for_each_model(lstm, rf, xg_reg, x_val, x_val_supervised)

    # get threshold for each model
    threshold_lstm = evaluation.find_best_threshold_fixed_fpr(y_val, y_val_pred_lstm.ravel())
    threshold_rf = evaluation.find_best_threshold_fixed_fpr(y_val, y_val_pred_rf[:, 1])
    threshold_xgb = evaluation.find_best_threshold_fixed_fpr(y_val, y_val_pred_xgb[:, 1])

    # predicting test set
    y_pred_lstm, y_pred_rf, y_pred_xgb = get_predictions_for_each_model(lstm, rf, xg_reg, x_test, x_test_supervised)
    y_pred_lstm = y_pred_lstm.ravel()
    y_pred_rf = y_pred_rf[:, 1].ravel()
    y_pred_xgb = y_pred_xgb[:, 1].ravel()

    y_pred_lstm = np.array(evaluation.adjusted_classes(y_pred_lstm, threshold_lstm))
    y_pred_rf = np.array(evaluation.adjusted_classes(y_pred_rf, threshold_rf))
    y_pred_xgb = np.array(evaluation.adjusted_classes(y_pred_xgb, threshold_xgb))

    y_test_pred = []

    lstm_indices_found = evaluation.get_indices(y_test, y_pred_lstm)
    rf_indices_found = evaluation.get_indices(y_test, y_pred_rf)
    xgboost_indices_found = evaluation.get_indices(y_test, y_pred_xgb)

    not_found_by_xgboost, not_found_by_rf, not_found_by_others = evaluation.get_jaccard_index(lstm_indices_found, rf_indices_found, xgboost_indices_found)

    for i in range(len(y_pred_lstm)):
        if y_pred_lstm[i] + y_pred_rf[i] + y_pred_xgb[i] >= 2:
            y_test_pred.append(1)
        else:
            y_test_pred.append(0)

    return y_test_pred, not_found_by_xgboost, not_found_by_rf, not_found_by_others

# for each model, get threshold and get mean of the output on validation set .
# on test set, calculate the distance from the calculated mean.
# The model that has the max distance will be the one who decides
def predict_test_based_on_more_confident(lstm, rf, xg_reg, x_val, x_val_supervised, y_val, x_test, x_test_supervised, y_test):
    y_val_pred_lstm, y_val_pred_rf, y_val_pred_xgb = get_predictions_for_each_model(lstm, rf, xg_reg, x_val, x_val_supervised)

    # get threshold for each model
    threshold_lstm = evaluation.find_best_threshold_fixed_fpr(y_val, y_val_pred_lstm.ravel())
    threshold_rf = evaluation.find_best_threshold_fixed_fpr(y_val, y_val_pred_rf[:, 1])
    threshold_xgb = evaluation.find_best_threshold_fixed_fpr(y_val, y_val_pred_xgb[:, 1])

    # get mean for each model
    lstm_mean = np.array(y_val_pred_lstm.ravel()).mean()
    rf_mean = np.array(y_val_pred_rf[:, 1]).mean()
    xgb_mean = np.array(y_val_pred_xgb[:, 1]).mean()

    # predicting test set
    y_pred_lstm, y_pred_rf, y_pred_xgb = get_predictions_for_each_model(lstm, rf, xg_reg, x_test, x_test_supervised)
    y_pred_lstm = y_pred_lstm.ravel()
    y_pred_rf = y_pred_rf[:, 1].ravel()
    y_pred_xgb = y_pred_xgb[:, 1].ravel()

    y_test_pred = []
    num_decisions_taken_by_lstm = 0
    num_decisions_correctly_taken_from_lstm = 0
    num_decisions_correctly_taken_from_lstm_and_not_from_xgb_or_rf = 0
    num_decisions_taken_by_rf = 0
    num_decisions_taken_by_xgb = 0
    for i in range(len(y_pred_lstm)):
        max_value = max(abs(y_pred_lstm[i] - lstm_mean), abs(y_pred_rf[i] - rf_mean), abs(y_pred_xgb[i] - xgb_mean))
        if abs(y_pred_lstm[i] - lstm_mean) == max_value:
            num_decisions_taken_by_lstm += 1
            lstm_output = 1 if y_pred_lstm[i] > threshold_lstm else 0
            rf_output = 1 if y_pred_rf[i] > threshold_rf else 0
            xgb_output = 1 if y_pred_xgb[i] > threshold_xgb else 0

            y_test_pred.append(lstm_output)
            if lstm_output == 1 and y_test[i] == 1 and (rf_output == 0 or xgb_output == 0):
                num_decisions_correctly_taken_from_lstm_and_not_from_xgb_or_rf += 1
            if lstm_output == 1 and y_test[i]:
                num_decisions_correctly_taken_from_lstm += 1

        elif abs(y_pred_rf[i] - rf_mean) == max_value:
            y_test_pred.append(1 if y_pred_rf[i] > threshold_rf else 0)
            num_decisions_taken_by_rf += 1
        else:
            num_decisions_taken_by_xgb += 1
            y_test_pred.append(1 if y_pred_xgb[i] > threshold_xgb else 0)

    print("Num decisions taken from lstm: ", num_decisions_taken_by_lstm)
    print("Num decisions taken by rf: ", num_decisions_taken_by_rf)
    print("Num decisions taken by xgb: ", num_decisions_taken_by_xgb)
    print("Num decisions taken by lstm correctly taken: ", num_decisions_correctly_taken_from_lstm)
    print("Num decisions taken by lstm correctly taken and not by others: ", num_decisions_correctly_taken_from_lstm_and_not_from_xgb_or_rf)
    return y_test_pred

# for each model, get threshold and get mean of the output on validation set .
# on test set, calculate the distance from the calculated mean.
# The model that has the max distance will be the one who decides only if his distance is greater than the sum of the other two distances.
# Otherwise, use the majority voting
def predict_test_based_on_more_confident_and_majority_voting(lstm, rf, xg_reg, x_val, x_val_supervised, y_val, x_test, x_test_supervised, y_test):
    y_val_pred_lstm, y_val_pred_rf, y_val_pred_xgb = get_predictions_for_each_model(lstm, rf, xg_reg, x_val, x_val_supervised)

    # get threshold for each model
    threshold_lstm = evaluation.find_best_threshold_fixed_fpr(y_val, y_val_pred_lstm.ravel())
    threshold_rf = evaluation.find_best_threshold_fixed_fpr(y_val, y_val_pred_rf[:, 1])
    threshold_xgb = evaluation.find_best_threshold_fixed_fpr(y_val, y_val_pred_xgb[:, 1])

    # get mean for each model
    lstm_mean = np.array(y_val_pred_lstm.ravel()).mean()
    rf_mean = np.array(y_val_pred_rf[:, 1]).mean()
    xgb_mean = np.array(y_val_pred_xgb[:, 1]).mean()

    # predicting test set
    y_pred_lstm, y_pred_rf, y_pred_xgb = get_predictions_for_each_model(lstm, rf, xg_reg, x_test, x_test_supervised)
    y_pred_lstm = y_pred_lstm.ravel()
    y_pred_rf = y_pred_rf[:, 1].ravel()
    y_pred_xgb = y_pred_xgb[:, 1].ravel()

    y_test_pred = []
    num_decisions_taken_by_lstm = 0
    num_decisions_correctly_taken_from_lstm = 0
    num_decisions_correctly_taken_from_lstm_and_not_from_xgb_or_rf = 0
    num_decisions_taken_by_rf = 0
    num_decisions_taken_by_xgb = 0
    for i in range(len(y_pred_lstm)):
        max_value = max(abs(y_pred_lstm[i] - lstm_mean), abs(y_pred_rf[i] - rf_mean), abs(y_pred_xgb[i] - xgb_mean))
        if abs(y_pred_lstm[i] - lstm_mean) == max_value:
            if max_value > abs(y_pred_rf[i] - rf_mean) + abs(y_pred_xgb[i] - xgb_mean):
                num_decisions_taken_by_lstm += 1
                lstm_output = 1 if y_pred_lstm[i] > threshold_lstm else 0
                rf_output = 1 if y_pred_rf[i] > threshold_rf else 0
                xgb_output = 1 if y_pred_xgb[i] > threshold_xgb else 0

                y_test_pred.append(lstm_output)
                if lstm_output == 1 and y_test[i] == 1 and (rf_output == 0 or xgb_output == 0):
                    num_decisions_correctly_taken_from_lstm_and_not_from_xgb_or_rf += 1
                if lstm_output == 1 and y_test[i]:
                    num_decisions_correctly_taken_from_lstm += 1
            else:
                if y_pred_lstm[i] + y_pred_rf[i] + y_pred_xgb[i] >= 2:
                    y_test_pred.append(1)
                else:
                    y_test_pred.append(0)

        elif abs(y_pred_rf[i] - rf_mean) == max_value:
            if max_value > abs(y_pred_lstm[i] - lstm_mean) + abs(y_pred_xgb[i] - xgb_mean):
                y_test_pred.append(1 if y_pred_rf[i] > threshold_rf else 0)
                num_decisions_taken_by_rf += 1
            else:
                if y_pred_lstm[i] + y_pred_rf[i] + y_pred_xgb[i] >= 2:
                    y_test_pred.append(1)
                else:
                    y_test_pred.append(0)
        else:
            if max_value > abs(y_pred_rf[i] - rf_mean) + abs(y_pred_lstm[i] - lstm_mean):
                num_decisions_taken_by_xgb += 1
                y_test_pred.append(1 if y_pred_xgb[i] > threshold_xgb else 0)
            else:
                if y_pred_lstm[i] + y_pred_rf[i] + y_pred_xgb[i] >= 2:
                    y_test_pred.append(1)
                else:
                    y_test_pred.append(0)

    print("Num decisions taken from lstm: ", num_decisions_taken_by_lstm)
    print("Num decisions taken by rf: ", num_decisions_taken_by_rf)
    print("Num decisions taken by xgb: ", num_decisions_taken_by_xgb)
    print("Num decisions taken by lstm correctly taken: ", num_decisions_correctly_taken_from_lstm)
    print("Num decisions taken by lstm correctly taken and not by others: ", num_decisions_correctly_taken_from_lstm_and_not_from_xgb_or_rf)
    return y_test_pred

def repeat_experiment_n_times(lstm, rf, xg_reg, scenario, times_to_repeat=100):
    tn_s = []
    tp_s = []
    fp_s = []
    fn_s = []
    f1_s = []
    balanced_accuracies = []
    precisions = []
    recalls = []
    aucpr_s = []
    roc_aucs = []
    not_found_by_xgboost, not_found_by_rf, not_found_by_others = 0, 0, 0

    for i in range(times_to_repeat):
        print("Iteration", i)
        x_test_set, y_test_set = sequences_crafting_for_classification.get_test_set(scenario=scenario)

        x_val, y_val, x_test, y_test = evaluation.get_val_test_set(x_test_set, y_test_set, val_size=0.25)
        x_val_supervised = x_val[:, len(x_val[0]) - 1, :]
        x_test_supervised = x_test[:, len(x_val[0]) - 1, :]

        try:
            # y_test_pred, not_by_xgb, not_by_rf, not_by_others = predict_test_based_on_voting(lstm, rf, xg_reg, x_val, x_val_supervised, y_val, x_test, x_test_supervised, y_test)
            y_test_pred = predict_test_based_on_more_confident(lstm, rf, xg_reg, x_val, x_val_supervised, y_val, x_test, x_test_supervised, y_test)
            # y_test_pred = predict_test_based_on_more_confident_and_majority_voting(lstm, rf, xg_reg, x_val, x_val_supervised, y_val, x_test, x_test_supervised, y_test)
            # not_found_by_xgboost += not_by_xgb
            # not_by_rf += not_by_rf
            # not_found_by_others += not_by_others

            y_test_pred = np.array(y_test_pred)
            confusion, f1, balanced_accuracy, precision, recall, aucpr, roc_auc = evaluation.get_performance(y_test, y_test_pred, threshold=True)
            tn = confusion[0, 0]
            tp = confusion[1, 1]
            fp = confusion[0, 1]
            fn = confusion[1, 0]

            tn_s.append(tn)
            tp_s.append(tp)
            fp_s.append(fp)
            fn_s.append(fn)
            f1_s.append(f1)
            balanced_accuracies.append(balanced_accuracy)
            precisions.append(precision)
            recalls.append(recall)
            aucpr_s.append(aucpr)
            roc_aucs.append(roc_auc)
        except RuntimeError:
            i -= 1

    print("Frauds found by lstm and not by xgboost:", not_found_by_xgboost/times_to_repeat)
    print("Frauds found by lstm and not by rf:", not_found_by_rf/times_to_repeat)
    print("Frauds found by lstm and not by the other models:", not_found_by_others/times_to_repeat)
    evaluation.print_results(np.array(tn_s).mean(), np.array(fp_s).mean(), np.array(fn_s).mean(), np.array(tp_s).mean(), np.array(f1_s).mean(), np.array(balanced_accuracies).mean(), np.array(precisions).mean(), np.array(recalls).mean(), np.array(aucpr_s).mean(), np.array(roc_aucs).mean())


look_back = LOOK_BACK
print("Lookback using: ", look_back)
x_train, y_train = sequences_crafting_for_classification.get_train_set()
# train model for supervised models (xgboost/rf)
x_train_supervised = x_train[:, look_back, :]
y_train_supervised = y_train

print("Training models...")
lstm = LSTM_classifier.create_fit_model(x_train, y_train, look_back)

best_lstm_model = {'layers': {'input': 128, 'hidden1': 32, 'output': 1}, 'epochs': 2, 'dropout_rate': 0.5, 'batch_size': 30}
# lstm = LSTM_classifier.create_model(best_lstm_model["layers"], best_lstm_model["dropout_rate"], len(x_train[0]) - 1, len(x_train_supervised[0]))
# lstm.fit(x_train, y_train, epochs=best_lstm_model["epochs"], batch_size=best_lstm_model["batch_size"])

best_params_xgboost = {'subsample': 0.8, 'min_child_weight': 5, 'max_depth': 5, 'gamma': 1.5, 'colsample_bytree': 0.6}
'''
xg_reg = xgb.XGBClassifier(subsample=best_params_xgboost["subsample"],
                           min_child_weight=best_params_xgboost["min_child_weight"],
                           max_depth=best_params_xgboost["max_depth"],
                           gamma=best_params_xgboost["gamma"],
                           colsample_bytree=best_params_xgboost["colsample_bytree"])
'''
xg_reg = xgb.XGBClassifier()
xg_reg.fit(x_train_supervised, y_train_supervised)

best_params_rf = {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_features': 'auto', 'max_depth': 10, 'bootstrap': True}
'''
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
    for scenario in scenarios:
        print("-------------------", scenario, "scenario --------------------------")
        repeat_experiment_n_times(lstm, rf, xg_reg, scenario, times_to_repeat=10)

if DATASET_TYPE == REAL_DATASET:
    # x_test_set, y_test_set = sequences_crafting_for_classification.create_test_set(look_back)
    repeat_experiment_n_times(lstm, rf, xg_reg, False, times_to_repeat=10)