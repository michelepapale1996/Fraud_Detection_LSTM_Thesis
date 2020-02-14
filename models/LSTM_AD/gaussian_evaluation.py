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
from scipy import stats

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

def mean_and_covariance(y, y_pred):
    errors = np.absolute(y - y_pred)
    return np.mean(errors, axis=0), np.std(errors, axis=0)

def get_probabilities(y, y_pred, distribution):
    return distribution.pdf(np.absolute(y[:, 0:len(y[0]) - 1] - y_pred))

# This function adjusts class predictions based on the prediction threshold (t).
def adjusted_classes(y_scores, t):
    return [1 if y >= t else 0 for y in y_scores]

def find_best_threshold(y_true, y_pred, distribution):
    probabilities = get_probabilities(y_true, y_pred, distribution)
    # thresholds = np.linspace(probabilities.min(), probabilities.max(), num=200, endpoint=True)
    thresholds = np.linspace(min(probabilities), max(probabilities), num=200, endpoint=True)
    scores = {}
    for t in thresholds:
        labels = adjusted_classes(probabilities, t)
        f = fbeta_score(y_true[:, len(y_true[0]) - 1], labels, beta=0.1)
        scores[t] = f
    return max(scores, key=scores.get)

def evaluate_model(model, x_val_no_frauds, y_val_no_frauds, x_val, y_val, x_test, y_test):
    # getting the mean and cov of the multivariate normal distribution

    y_val_pred_no_frauds = model.predict(x_val_no_frauds)
    mean, cov = mean_and_covariance(y_val_no_frauds, y_val_pred_no_frauds)
    distribution = stats.multivariate_normal(mean=mean, cov=cov)

    y_val_pred = model.predict(x_val)
    best_threshold = find_best_threshold(y_val, y_val_pred, distribution)

    y_test_pred = model.predict(x_test)

    probabilities = get_probabilities(y_test, y_test_pred, distribution)

    labels = adjusted_classes(probabilities, best_threshold)
    return evaluate(y_test[:, len(y_test[0]) - 1], labels, best_threshold)