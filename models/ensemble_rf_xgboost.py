from models import LSTM_classifier, evaluation, RF, xgboost_classifier
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from dataset_creation import sequences_crafting_for_classification
from dataset_creation.constants import *
from sklearn.metrics import f1_score, recall_score, precision_score, balanced_accuracy_score, confusion_matrix
from scipy import stats

def get_predictions_for_each_model(rf, xg_reg, x_supervised):
    y_rf = rf.predict_proba(x_supervised)
    y_xgb = xg_reg.predict_proba(x_supervised)
    return y_rf[:, 1], y_xgb[:,1]

# for each model, get threshold and get mean of the output on validation set .
# on test set, calculate the distance from the calculated mean.
# The model that has the max distance will be the one who decides
def predict_test_based_on_more_confident(rf, xg_reg, x_val_supervised, y_val, x_test_supervised):
    y_val_pred_rf, y_val_pred_xgb = get_predictions_for_each_model(rf, xg_reg, x_val_supervised)

    # get threshold for each model
    threshold_rf = evaluation.find_best_threshold_fixed_fpr(y_val, y_val_pred_rf[:, 1])
    threshold_xgb = evaluation.find_best_threshold_fixed_fpr(y_val, y_val_pred_xgb[:, 1])

    # get mean for each model
    rf_mean = np.array(y_val_pred_rf[:, 1]).mean()
    xgb_mean = np.array(y_val_pred_xgb[:, 1]).mean()

    # get std for each model
    rf_std = np.array(y_val_pred_rf[:, 1]).std()
    xgb_std = np.array(y_val_pred_xgb[:, 1]).std()

    # predicting test set
    y_pred_rf, y_pred_xgb = get_predictions_for_each_model(rf, xg_reg, x_test_supervised)
    y_pred_rf = y_pred_rf[:, 1].ravel()
    y_pred_xgb = y_pred_xgb[:, 1].ravel()

    y_test_pred = []
    for i in range(len(y_pred_rf)):
        max_value = max(abs(y_pred_rf[i] - rf_mean) / rf_std, abs(y_pred_xgb[i] - xgb_mean) / xgb_std)
        if abs(y_pred_rf[i] - rf_mean) / rf_std == max_value:
            y_test_pred.append(1 if y_pred_rf[i] > threshold_rf else 0)
        else:
            y_test_pred.append(1 if y_pred_xgb[i] > threshold_xgb else 0)

    return y_test_pred

def predict_test_based_on_exponential(rf, xg_reg, x_val_supervised, y_val, x_test_supervised):
    y_val_pred_rf, y_val_pred_xgb = get_predictions_for_each_model(rf, xg_reg, x_val_supervised)

    # for each model, fit the exponential distribution
    loc_rf, scale_rf = stats.expon.fit(y_val_pred_rf)
    loc_xgb, scale_xgb = stats.expon.fit(y_val_pred_xgb)
    samples_rf = stats.expon.cdf(y_val_pred_rf, scale=scale_rf, loc=loc_rf)
    samples_xgb = stats.expon.cdf(y_val_pred_xgb, scale=scale_xgb, loc=loc_xgb)

    # get threshold for each model
    threshold_rf = evaluation.find_best_threshold_fixed_fpr(y_val, samples_rf)
    threshold_xgb = evaluation.find_best_threshold_fixed_fpr(y_val, samples_xgb)

    # get mean for each model
    rf_mean = np.array(samples_rf).mean()
    xgb_mean = np.array(samples_xgb).mean()
    # get std for each model
    rf_std = np.array(samples_rf).std()
    xgb_std = np.array(samples_xgb).std()

    # predicting test set
    y_pred_rf, y_pred_xgb = get_predictions_for_each_model(rf, xg_reg, x_test_supervised)
    samples_rf = stats.expon.cdf(y_pred_rf, scale=scale_rf, loc=loc_rf)
    samples_xgb = stats.expon.cdf(y_pred_xgb, scale=scale_xgb, loc=loc_xgb)

    y_test_pred = []
    for i in range(len(samples_rf)):
        max_value = max(abs(samples_rf[i] - rf_mean) / rf_std, abs(samples_xgb[i] - xgb_mean) / xgb_std)
        if abs(samples_rf[i] - rf_mean) / rf_std == max_value:
            y_test_pred.append(1 if samples_rf[i] > threshold_rf else 0)
        else:
            y_test_pred.append(1 if samples_xgb[i] > threshold_xgb else 0)

    return y_test_pred

# each model gives his opinion and the output is based on the majority
def predict_test_based_on_voting(rf, xg_reg, x_val_supervised, y_val, x_test_supervised):
    y_val_pred_rf, y_val_pred_xgb = get_predictions_for_each_model(rf, xg_reg, x_val_supervised)

    # get threshold for each model
    threshold_rf = evaluation.find_best_threshold_fixed_fpr(y_val, y_val_pred_rf[:, 1])
    threshold_xgb = evaluation.find_best_threshold_fixed_fpr(y_val, y_val_pred_xgb[:, 1])

    # predicting test set
    y_pred_rf, y_pred_xgb = get_predictions_for_each_model(rf, xg_reg, x_test_supervised)
    y_pred_rf = y_pred_rf[:, 1].ravel()
    y_pred_xgb = y_pred_xgb[:, 1].ravel()

    y_pred_rf = evaluation.adjusted_classes(y_pred_rf, threshold_rf)
    y_pred_xgb = evaluation.adjusted_classes(y_pred_xgb, threshold_xgb)

    y_test_pred = []
    for i in range(len(y_pred_rf)):
        if y_pred_rf[i] + y_pred_xgb[i] >= 2:
            y_test_pred.append(1)
        else:
            y_test_pred.append(0)

    return y_test_pred


look_back = LOOK_BACK
print("Lookback using: ", look_back)
x_train, y_train = sequences_crafting_for_classification.get_train_set()

x_train_supervised = x_train[:, look_back, :]
y_train_supervised = y_train

xg_reg = xgboost_classifier.create_model(x_train_supervised, y_train_supervised)
rf = RF.create_model(x_train_supervised, y_train_supervised)

times_to_repeat = 100
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
x_test_set, y_test_set = sequences_crafting_for_classification.get_test_set()

for i in range(times_to_repeat):
    x_val, y_val, x_test, y_test = evaluation.get_val_test_set(x_test_set, y_test_set)
    x_val_supervised = x_val[:, look_back, :]
    x_test_supervised = x_test[:, look_back, :]

    try:
        # y_test_pred = predict_test_based_on_voting(rf, xg_reg, x_val_supervised, y_val, x_test_supervised)
        # y_test_pred = predict_test_based_on_more_confident(rf, xg_reg, x_val_supervised, y_val, x_test_supervised)
        y_test_pred = predict_test_based_on_exponential(rf, xg_reg, x_val_supervised, y_val, x_test_supervised)
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
evaluation.print_results(np.array(tn_s).mean(), np.array(fp_s).mean(), np.array(fn_s).mean(), np.array(tp_s).mean(), np.array(f1_s).mean(), np.array(balanced_accuracies).mean(), np.array(precisions).mean(), np.array(recalls).mean(), np.array(aucpr_s).mean(), np.array(roc_aucs).mean())