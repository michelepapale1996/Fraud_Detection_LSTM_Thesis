import numpy as np
from sklearn.metrics import f1_score, \
                            average_precision_score, \
                            roc_auc_score, \
                            precision_recall_curve, \
                            auc, \
                            mean_absolute_error, \
                            confusion_matrix, \
                            accuracy_score, \
                            balanced_accuracy_score, \
                            fbeta_score, precision_score, recall_score

def get_errors(y_true, y_pred):
    len_y = len(y_pred)
    errors = []
    for i in range(len_y):
        error = mean_absolute_error(y_true[i], y_pred[i])
        errors.append(error)
    return errors

# This function adjusts class predictions based on the prediction threshold (t).
# todo: use builtin sklearn function
def adjusted_classes(y_scores, t):
    return [1 if y >= t else 0 for y in y_scores]

def find_best_threshold(y_true, y_pred):
    probabilities = get_errors(y_true[:, 0:len(y_true[0]) - 1], y_pred)
    thresholds = np.linspace(min(probabilities), max(probabilities), num=500, endpoint=True)
    scores = {}
    for t in thresholds:
        labels = adjusted_classes(probabilities, t)
        f = fbeta_score(y_true[:,len(y_true[0]) - 1], labels, beta=0.1)
        scores[t] = f
    return max(scores, key=scores.get)

def evaluate(y_true, y_pred, threshold):
    roc_auc = roc_auc_score(y_true, y_pred)
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    precision_recall_auc = auc(recall, precision)
    average_precision = average_precision_score(y_true, y_pred)

    y_pred = adjusted_classes(y_pred, threshold)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    fbeta = fbeta_score(y_true, y_pred, beta=0.1)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)
    return f1, fbeta, accuracy, precision, recall, balanced_accuracy, confusion, roc_auc, precision_recall_auc, average_precision

def evaluate_model(model, x_val, y_val, x_test, y_test):
    # using validation set to find the best threshold to divide frauds and genuine
    y_val_pred = model.predict(x_val[:,:, 0: x_val.shape[2] - 1])
    best_threshold = find_best_threshold(y_val, y_val_pred)

    y_test_pred = model.predict(x_test[:,:, 0: x_val.shape[2] - 1])
    probabilities = get_errors(y_test[:, 0:len(y_test[0]) - 1], y_test_pred)
    labels = adjusted_classes(probabilities, best_threshold)
    return evaluate(y_test[:, len(y_test[0]) - 1], labels, best_threshold)