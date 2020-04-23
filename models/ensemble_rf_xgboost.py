from models import evaluation, RF, xgboost_classifier, resampling_dataset
import numpy as np
from dataset_creation import sequences_crafting_for_classification
from dataset_creation.constants import *
from scipy import stats

def get_predictions_for_each_model(rf, xg_reg, x_supervised):
    y_rf = rf.predict_proba(x_supervised)
    y_xgb = xg_reg.predict_proba(x_supervised)
    return y_rf[:, 1], y_xgb[:,1]

# for each model, get threshold and get mean of the output on validation set .
# on test set, calculate the distance from the calculated mean.
# The model that has the max distance will be the one who decides
def predict_test_based_on_more_confident(rf, xg_reg, x_test_supervised, threshold_rf, threshold_xgb, min_rf, min_xgb, max_rf, max_xgb, mean_rf, mean_xgb):
    def min_max_rescaling(values, min_, max_):
        return [(ith_element - min_) / (max_ - min_) for ith_element in values]
    # predicting test set
    y_pred_rf, y_pred_xgb = get_predictions_for_each_model(rf, xg_reg, x_test_supervised)
    y_pred_rf = y_pred_rf.ravel()
    y_pred_xgb = y_pred_xgb.ravel()

    y_pred_rf = min_max_rescaling(y_pred_rf, min_rf, max_rf)
    y_pred_xgb = min_max_rescaling(y_pred_xgb, min_xgb, max_xgb)

    y_test_pred = []
    for i in range(len(y_pred_rf)):
        max_value = max(abs(y_pred_rf[i] - mean_rf), abs(y_pred_xgb[i] - mean_xgb))
        if abs(y_pred_rf[i] - mean_rf) == max_value:
            y_test_pred.append(1 if y_pred_rf[i] > threshold_rf else 0)
        else:
            y_test_pred.append(1 if y_pred_xgb[i] > threshold_xgb else 0)

    return y_test_pred

def predict_test_based_on_exponential(rf, scale_rf, loc_rf, mean_rf, std_rf, threshold_rf, xg_reg, scale_xgb, loc_xgb, mean_xgb, std_xgb, threshold_xgb, x_test_supervised):
    # predicting test set
    y_pred_rf, y_pred_xgb = get_predictions_for_each_model(rf, xg_reg, x_test_supervised)
    samples_rf = stats.expon.cdf(y_pred_rf, scale=scale_rf, loc=loc_rf)
    samples_xgb = stats.expon.cdf(y_pred_xgb, scale=scale_xgb, loc=loc_xgb)

    y_test_pred = []
    thresholds = []
    for i in range(len(samples_rf)):
        max_value = max(abs(samples_rf[i] - mean_rf) / std_rf, abs(samples_xgb[i] - mean_xgb) / std_xgb)
        if abs(samples_rf[i] - mean_rf) / std_rf == max_value:
            y_test_pred.append(samples_rf[i])
            thresholds.append(threshold_rf)
        else:
            y_test_pred.append(samples_xgb[i])
            thresholds.append(threshold_xgb)

    return y_test_pred, thresholds


look_back = LOOK_BACK
print("Lookback using: ", look_back)
x_train, y_train = sequences_crafting_for_classification.get_train_set()

# if the dataset is the real one -> contrast imbalanced dataset problem
if DATASET_TYPE == REAL_DATASET:
    x_train, y_train = resampling_dataset.oversample_set(x_train, y_train)

x_train_supervised = x_train[:, look_back, :]
y_train_supervised = y_train

times_to_repeat = 10
rf, scale_rf, loc_rf, mean_rf, std_rf, threshold_rf = RF.create_fit_model_for_ensemble_based_on_cdf(x_train_supervised, y_train_supervised, times_to_repeat=times_to_repeat)
xg_reg, scale_xgb, loc_xgb, mean_xgb, std_xgb, threshold_xgb = xgboost_classifier.create_fit_model_for_ensemble_based_on_cdf(x_train_supervised, y_train_supervised, times_to_repeat=times_to_repeat)

x_test, y_test = sequences_crafting_for_classification.get_test_set()
x_test_supervised = x_test[:, look_back, :]

y_test_pred, thresholds = predict_test_based_on_exponential(rf, scale_rf, loc_rf, mean_rf, std_rf, threshold_rf, xg_reg, scale_xgb, loc_xgb, mean_xgb, std_xgb, threshold_xgb, x_test_supervised)
y_test_pred = np.array(y_test_pred)
confusion, f1, balanced_accuracy, precision, recall, aucpr, roc_auc, fpr_values, tpr_values, accuracy, matthews_coeff = evaluation.get_performance(y_test, y_test_pred, thresholds)
tn = confusion[0, 0]
tp = confusion[1, 1]
fp = confusion[0, 1]
fn = confusion[1, 0]

evaluation.print_results(tn, fp, fn, tp, f1, balanced_accuracy, precision, recall, aucpr, roc_auc, fpr_values, tpr_values, accuracy, matthews_coeff)