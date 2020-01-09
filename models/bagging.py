from sklearn.ensemble import BaggingClassifier
from models import evaluation
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


look_back = LOOK_BACK
print("Lookback using: ", look_back)
x_train, y_train = sequences_crafting_for_classification.create_train_set(look_back)
# train model for supervised models (xgboost/rf)
x_train_supervised = x_train[:, look_back, :]
y_train_supervised = y_train

print("Fitting model...")
clf_xgb = BaggingClassifier(base_estimator=xgb.XGBClassifier(), n_estimators=5, random_state=0).fit(x_train_supervised, y_train_supervised)

clf_rf = BaggingClassifier(base_estimator=RandomForestClassifier(), n_estimators=5, random_state=0).fit(x_train_supervised, y_train_supervised)
print("Testing model...")
x_test, y_test = sequences_crafting_for_classification.create_test_set(look_back)
# train model for supervised models (xgboost/rf)
x_test_supervised = x_test[:, look_back, :]
y_test_supervised = y_test

y_test_pred_rf = clf_rf.predict_proba(x_test_supervised)
y_test_pred_xgb = clf_xgb.predict_proba(x_test_supervised)

evaluation.evaluate(y_test_supervised, y_test_pred_rf[:,1] + y_test_pred_xgb[:,1])