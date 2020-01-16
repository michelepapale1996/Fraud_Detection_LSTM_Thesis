from models import LSTM_classifier, evaluation
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from dataset_creation import sequences_crafting_for_classification
from dataset_creation.constants import *
from sklearn.metrics import f1_score, recall_score, precision_score, balanced_accuracy_score, confusion_matrix
import random

def get_predictions_for_each_model(rf, xg_reg, x_supervised):
    y_rf = rf.predict_proba(x_supervised)
    y_xgb = xg_reg.predict_proba(x_supervised)
    return y_rf, y_xgb

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
# x_train, y_train = sequences_crafting_for_classification.create_train_set(look_back)
x_train = np.load("/home/mpapale/thesis/classification/dataset_" + str(look_back) + "_lookback/x_train_10_trx_per_user.npy")
y_train = np.load("/home/mpapale/thesis/classification/dataset_" + str(look_back) + "_lookback/y_train_10_trx_per_user.npy")

x_train_supervised = x_train[:, look_back, :]
y_train_supervised = y_train

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
# x_test_set, y_test_set = sequences_crafting_for_classification.create_test_set(look_back)
x_test_set = np.load("/home/mpapale/thesis/classification/dataset_" + str(look_back) + "_lookback/x_test_10_trx_per_user.npy")
y_test_set = np.load("/home/mpapale/thesis/classification/dataset_" + str(look_back) + "_lookback/y_test_10_trx_per_user.npy")

for i in range(times_to_repeat):
    x_val, y_val, x_test, y_test = evaluation.get_val_test_set(x_test_set, y_test_set)
    x_val_supervised = x_val[:, look_back, :]
    x_test_supervised = x_test[:, look_back, :]

    try:
        y_test_pred = predict_test_based_on_voting(rf, xg_reg, x_val_supervised, y_val, x_test_supervised)
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