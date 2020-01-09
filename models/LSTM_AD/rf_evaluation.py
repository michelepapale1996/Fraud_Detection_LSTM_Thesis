import numpy as np
from sklearn.ensemble import RandomForestClassifier
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

# This function adjusts class predictions based on the prediction threshold (t).
# todo: use builtin sklearn function
def adjusted_classes(y_scores, t):
    return [1 if y >= t else 0 for y in y_scores]

def evaluate(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    fbeta = fbeta_score(y_true, y_pred, beta=0.1)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)
    return f1, fbeta, accuracy, precision, recall, balanced_accuracy, confusion

def get_errors(y_true, y_pred):
    len_y = len(y_pred)
    errors = []
    for i in range(len_y):
        error = mean_absolute_error(y_true[i], y_pred[i])
        errors.append(error)
    return errors

def evaluate_model(model, x_val, y_val, x_test, y_test):
    # for each trx, get the mae of each feature
    y_val_pred = model.predict(x_val)
    errors = abs(y_val[:, 0:len(y_val[0]) - 1] - y_val_pred)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(errors, y_val[:, len(y_val[0]) - 1])

    y_test_pred = model.predict(x_test)
    errors = abs(y_test[:, 0:len(y_test[0]) - 1] - y_test_pred)
    predicted_labels = rf.predict(errors)
    return evaluate(y_test[:, len(y_test[0]) - 1], predicted_labels)