from sklearn.metrics import f1_score, recall_score, precision_score, roc_curve, roc_auc_score, precision_recall_curve, \
    auc, balanced_accuracy_score, confusion_matrix, accuracy_score, jaccard_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import random

def get_position(recall):
    for i in range(0, len(recall)):
        if recall[i] < 0.2:
            return i - 1

# This function adjusts class predictions based on the prediction threshold (t).
def adjusted_classes(y_scores, t):
    return [1 if y >= t else 0 for y in y_scores]

def find_best_threshold_based_on_recall(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    position = get_position(recall)
    return thresholds[position]

# The best cut-off point is the one that maximizes J = TPR - FPR
def find_best_threshold_based_on_youden_j_statistic(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    max_scores = []
    for index in range(len(thresholds)):
        max_scores.append(tpr[index] - fpr[index])
    index_best_threshold = max_scores.index(max(max_scores))
    # print("best threshold: ", thresholds[index_best_threshold])
    return thresholds[index_best_threshold]

# cost function is the function used in FraudHunter:
# C = (FP + cost_ratio * FN) / (FP + TP + cost_ratio * (FN + TN))
def find_best_threshold_that_minimizes_cost_function(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    scores = []
    for index in range(len(thresholds)):
        y_pred = adjusted_classes(y_pred, thresholds[index])
        c = confusion_matrix(y_true, y_pred)
        tn = c[0, 0]
        tp = c[1, 1]
        fp = c[0, 1]
        fn = c[1, 0]
        scores.append( (fp + 100 * fn) / (fp + tp + 100 * (fn + tn)))
    # print("Scores:", scores)
    index_best_threshold = scores.index(min(scores))
    return thresholds[index_best_threshold]


# find the best threshold maximizing tpr up to a fpr of max_fpr_rate
def find_best_threshold_fixed_fpr(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    MAX_FPR_RATE = 0.3
    if len(tpr) == 0:
        raise RuntimeError("tpr is empty")
    best_tpr = tpr[0]

    for i in range(0, len(fpr)):
        if fpr[i] <= MAX_FPR_RATE:
            best_tpr = max(best_tpr, tpr[i])
        else:
            break
    if np.isnan(best_tpr):
        raise RuntimeError("Tpr cannot be nan")
    index_best_threshold = tpr.tolist().index(best_tpr)
    return thresholds[index_best_threshold]

def get_val_test_indices(y_test, val_size=0.25):
    fraud_indices = np.where(y_test == 1)
    genuine_indices = np.where(y_test == 0)
    val_fraud_indices = random.sample(fraud_indices[0].tolist(), int(len(fraud_indices[0]) * val_size))
    test_fraud_indices = list(set(fraud_indices[0]) - set(val_fraud_indices))

    val_genuine_indices = random.sample(genuine_indices[0].tolist(), int(len(genuine_indices[0]) * val_size))
    test_genuine_indices = list(set(genuine_indices[0]) - set(val_genuine_indices))

    val_indices = val_fraud_indices + val_genuine_indices
    test_indices = test_fraud_indices + test_genuine_indices

    return val_indices, test_indices

def get_val_test_set(x_test, y_test, val_size=0.25):
    val_indices, test_indices = get_val_test_indices(y_test, val_size)

    x_val, y_val = x_test[val_indices], y_test[val_indices]
    x_test, y_test = x_test[test_indices], y_test[test_indices]
    return x_val, y_val, x_test, y_test

def get_metrics(y_true, y_pred):
    c = confusion_matrix(y_true, y_pred)
    f = f1_score(y_true, y_pred)
    b = balanced_accuracy_score(y_true, y_pred)
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return c, f, b, p, r

# if threshold is False, divides test set in test set and validation set to find the best threshold
def get_performance(y_true, y_pred, val_size=0.25, threshold=False):
    if not threshold:
        val_indices, test_indices = get_val_test_indices(y_true, val_size)
        # threshold = find_best_threshold_based_on_recall(y_true[val_indices], y_pred[val_indices])
        # threshold = find_best_threshold_based_on_youden_j_statistic(y_true[val_indices], y_pred[val_indices])
        # threshold = find_best_threshold_that_minimizes_cost_function(y_true[val_indices], y_pred[val_indices])
        threshold = find_best_threshold_fixed_fpr(y_true[val_indices], y_pred[val_indices])

        y_true = y_true[test_indices]
        y_pred = y_pred[test_indices]
        y_pred = adjusted_classes(y_pred, threshold)

    roc_auc = roc_auc_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    y_pred = adjusted_classes(y_pred, threshold)

    c, f, b, p, r = get_metrics(y_true, y_pred)
    aucpr = auc(recall, precision, reorder=True)
    return c, f, b, p, r, aucpr, roc_auc

def print_results(tn, fp, fn, tp, f1, balanced_accuracy, precision, recall, aucpr, roc_auc):
    print("Confusion Matrix:")
    print(tn, fp)
    print(fn, tp)
    print("f1: ", f1)
    print("balanced accuracy: ", balanced_accuracy)
    print("precision: ", precision)
    print("recall (tpr): ", recall)
    print("fpr: ", fp / (fp + tn))
    print("aucpr: ", aucpr)
    print("Roc_auc score:", roc_auc)

# takes real and predicted output of a model and print results.
# If the threshold is set to False, then sets are divided in val and test set to find the right threshold
# If the threshold is set, then real and predicted are used only for testing.
def evaluate_n_times(real, predicted, times_to_repeat=100, threshold=False):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    f1_scores = 0
    balanced_accuracy_scores = 0
    precision_scores = 0
    recall_scores = 0
    aucpr_scores = 0
    roc_auc_scores = 0
    for i in range(times_to_repeat):
        confusion, f1, balanced_accuracy, precision, recall, aucpr, roc_auc = get_performance(real, predicted, threshold=threshold)
        tn += confusion[0, 0]
        tp += confusion[1, 1]
        fp += confusion[0, 1]
        fn += confusion[1, 0]
        f1_scores += f1
        balanced_accuracy_scores += balanced_accuracy
        precision_scores += precision
        recall_scores += recall
        aucpr_scores += aucpr
        roc_auc_scores += roc_auc

    print_results(tn / times_to_repeat,
                  fp / times_to_repeat,
                  fn / times_to_repeat,
                  tp / times_to_repeat,
                  f1_scores / times_to_repeat,
                  balanced_accuracy_scores / times_to_repeat,
                  precision_scores / times_to_repeat,
                  recall_scores / times_to_repeat,
                  aucpr_scores / times_to_repeat,
                  roc_auc_scores / times_to_repeat)
    return f1_scores / times_to_repeat, roc_auc_scores / times_to_repeat

def get_indices(y_test, y_pred_test):
    indices = set(np.where(y_pred_test == 1)[0])
    real_indices = set(np.where(y_test == 1)[0])
    indices_found = indices.intersection(real_indices)
    return indices_found

def get_found_frauds_indices(y_val_pred, y_test_pred, y_val, y_test):
    threshold = find_best_threshold_fixed_fpr(y_val, y_val_pred)
    y_pred_test = np.array(adjusted_classes(y_test_pred, threshold))
    return get_indices(y_test, y_pred_test)

def get_jaccard_index(lstm_indices, rf_indices, xgboost_indices):
    not_found_by_xgboost = len(lstm_indices - xgboost_indices)
    not_found_by_rf = len(lstm_indices - rf_indices)
    not_found_by_others = len(lstm_indices - rf_indices - xgboost_indices)

    print("Frauds found by lstm and not by xgboost:", not_found_by_xgboost)
    print("Frauds found by lstm and not by rf:", not_found_by_rf)
    print("Frauds found by lstm and not by the other models:", not_found_by_others)

    return not_found_by_xgboost, not_found_by_rf, not_found_by_others

def print_jaccard_index(lstm, rf, xg_reg, x_test, y_test, look_back):
    x_val, y_val, x_test, y_test = get_val_test_set(x_test, y_test)
    x_val_supervised = x_val[:, look_back, :]
    x_test_supervised = x_test[:, look_back, :]

    y_pred_val = lstm.predict(x_val)
    y_pred_test = lstm.predict(x_test)
    lstm_indices_found = get_found_frauds_indices(y_pred_val, y_pred_test, y_val, y_test)

    y_pred_val = rf.predict_proba(x_val_supervised)
    y_pred_test = rf.predict_proba(x_test_supervised)
    rf_indices_found = get_found_frauds_indices(y_pred_val[:, 1], y_pred_test[:, 1], y_val, y_test)

    y_pred_val = xg_reg.predict_proba(x_val_supervised)
    y_pred_test = xg_reg.predict_proba(x_test_supervised)
    xgboost_indices_found = get_found_frauds_indices(y_pred_val[:, 1], y_pred_test[:, 1], y_val, y_test)

    return get_jaccard_index(lstm_indices_found, rf_indices_found, xgboost_indices_found)