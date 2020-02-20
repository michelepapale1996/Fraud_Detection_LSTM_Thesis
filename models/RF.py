import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.append("/home/mpapale/thesis")
from dataset_creation import sequences_crafting_for_classification, constants
from models import LSTM_classifier, evaluation, resampling_dataset, explainability
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


def create_model(x_train, y_train, params=None):
    if not params:
        if constants.USING_AGGREGATED_FEATURES:
            if constants.DATASET_TYPE == constants.INJECTED_DATASET:
                params = constants.BEST_PARAMS_RF_AGGREGATED
            if constants.DATASET_TYPE == constants.REAL_DATASET:
                params = constants.BEST_PARAMS_RF_REAL_DATASET_AGGREGATED
            if constants.DATASET_TYPE == constants.OLD_DATASET:
                params = constants.BEST_PARAMS_RF_OLD_DATASET_AGGREGATED
        else:
            if constants.DATASET_TYPE == constants.INJECTED_DATASET:
                params = constants.BEST_PARAMS_RF_NO_AGGREGATED
            if constants.DATASET_TYPE == constants.REAL_DATASET:
                params = constants.BEST_PARAMS_RF_REAL_DATASET_NO_AGGREGATED
            if constants.DATASET_TYPE == constants.OLD_DATASET:
                params = constants.BEST_PARAMS_RF_OLD_DATASET_NO_AGGREGATED

    rf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                min_samples_split=params["min_samples_split"],
                                min_samples_leaf=params["min_samples_leaf"],
                                max_features=params["max_features"],
                                max_depth=params["max_depth"],
                                bootstrap=params["bootstrap"])
    rf.fit(x_train, y_train)
    return rf

def model_selection(x_train, y_train):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=5)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 100, num=5)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [5, 10, 20]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 4, 10]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf = RandomForestClassifier()
    rf_random = GridSearchCV(estimator=rf, param_grid=random_grid, cv=3, scoring="roc_auc", verbose=3, n_jobs=-1)

    # Fit the random search model
    rf_random.fit(x_train, y_train)

    print("Best params: ", rf_random.best_params_)
    best_random = rf_random.best_estimator_
    return best_random


if __name__ == "__main__":
    look_back = constants.LOOK_BACK

    x_train, y_train = sequences_crafting_for_classification.get_train_set()
    x_test, y_test = sequences_crafting_for_classification.get_test_set()

    # if the dataset is the real one -> contrast imbalanced dataset problem
    if constants.DATASET_TYPE == constants.REAL_DATASET:
        x_train, y_train = resampling_dataset.oversample_set(x_train, y_train)

    # adapt train and test set to supervised learning without time windows
    x_train_sup = x_train[:, look_back, :]
    x_test_sup = x_test[:, look_back, :]

    print("Fitting model...")
    # model = create_model(x_train_sup, y_train)
    model = model_selection(x_train_sup, y_train)

    print("Evaluating model...")
    y_pred = model.predict_proba(x_test_sup)
    y_pred = y_pred[:, 1]
    evaluation.evaluate_n_times(y_test, y_pred)
    # explainability.explain_dataset(model, x_train_sup, x_test_sup, y_test)