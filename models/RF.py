import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.append("/home/mpapale/thesis")
from dataset_creation import sequences_crafting_for_classification, constants
from models import LSTM_classifier, evaluation, resampling_dataset, explainability
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy import stats

def create_model(x_train, y_train, params=None, times_to_repeat=1):
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

    avg_threshold = 0
    for i in range(times_to_repeat):
        print("iteration: ", i, "of", times_to_repeat)
        _x_val, _y_val, _x_train, _y_train = evaluation.get_val_test_set(x_train, y_train, 0.25)

        rf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                    min_samples_split=params["min_samples_split"],
                                    min_samples_leaf=params["min_samples_leaf"],
                                    max_features=params["max_features"],
                                    max_depth=params["max_depth"],
                                    bootstrap=params["bootstrap"])
        rf.fit(_x_train, _y_train)
        y_pred = rf.predict_proba(_x_val)[:, 1]
        avg_threshold += evaluation.find_best_threshold_fixed_fpr(_y_val, y_pred)

    rf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                min_samples_split=params["min_samples_split"],
                                min_samples_leaf=params["min_samples_leaf"],
                                max_features=params["max_features"],
                                max_depth=params["max_depth"],
                                bootstrap=params["bootstrap"])
    rf.fit(x_train, y_train)
    return rf, avg_threshold / times_to_repeat

def create_model_for_ensemble(x_train, y_train, params=None, times_to_repeat=1):
    def min_max_rescaling(values, min_, max_):
        return [(ith_element - min_) / (max_ - min_) for ith_element in values]

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

    avg_threshold = []
    avg_min = []
    avg_max = []
    avg_mean = []
    for i in range(times_to_repeat):
        print("iteration: ", i, "of", times_to_repeat)
        _x_val, _y_val, _x_train, _y_train = evaluation.get_val_test_set(x_train, y_train, 0.25)

        rf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                    min_samples_split=params["min_samples_split"],
                                    min_samples_leaf=params["min_samples_leaf"],
                                    max_features=params["max_features"],
                                    max_depth=params["max_depth"],
                                    bootstrap=params["bootstrap"])
        rf.fit(_x_train, _y_train)
        y_pred = rf.predict_proba(_x_val)[:, 1]
        min_ = min(y_pred)
        max_ = max(y_pred)
        mean_ = np.array(y_pred).mean()
        avg_min.append(min_)
        avg_max.append(max_)
        avg_mean.append(mean_)

        y_pred = min_max_rescaling(y_pred, min_, max_)
        avg_threshold.append(evaluation.find_best_threshold_fixed_fpr(_y_val, y_pred))

    rf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                min_samples_split=params["min_samples_split"],
                                min_samples_leaf=params["min_samples_leaf"],
                                max_features=params["max_features"],
                                max_depth=params["max_depth"],
                                bootstrap=params["bootstrap"])
    rf.fit(x_train, y_train)
    return rf, np.array(avg_threshold).mean(), np.array(avg_min).mean(), np.array(avg_max).mean(), np.array(avg_mean).mean()

def create_fit_model_for_ensemble_based_on_cdf(x_train, y_train, params=None, times_to_repeat=1):
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

    scales, locs, means, stds, thresholds = [], [], [], [], []
    for i in range(times_to_repeat):
        print("iteration: ", i, "of", times_to_repeat)

        _x_val, _y_val, _x_train, _y_train = evaluation.get_val_test_set(x_train, y_train, 0.25)
        model = RandomForestClassifier(n_estimators=params["n_estimators"],
                                    min_samples_split=params["min_samples_split"],
                                    min_samples_leaf=params["min_samples_leaf"],
                                    max_features=params["max_features"],
                                    max_depth=params["max_depth"],
                                    bootstrap=params["bootstrap"])
        model.fit(_x_train, _y_train)
        y_val_pred_lstm = model.predict_proba(_x_val)[:, 1]

        '''
        to get the shape of the distribution
        prova = y_val_pred_lstm.tolist()
        buckets = {}
        for i in range(0, 101):
            buckets[i / 100] = 0
        for i in prova:
            buckets[round(i, 2)] += 1
        plt.plot(list(buckets.keys()), ",", list(buckets.values())
        plt.show()
        '''

        loc, scale = stats.expon.fit(y_val_pred_lstm)
        samples = stats.expon.cdf(y_val_pred_lstm, scale=scale, loc=loc)
        mean = samples.mean()
        std = samples.std()

        threshold = evaluation.find_best_threshold_fixed_fpr(_y_val, samples)
        scales.append(scale)
        locs.append(loc)
        means.append(mean)
        stds.append(std)
        thresholds.append(threshold)
    rf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                min_samples_split=params["min_samples_split"],
                                min_samples_leaf=params["min_samples_leaf"],
                                max_features=params["max_features"],
                                max_depth=params["max_depth"],
                                bootstrap=params["bootstrap"])
    rf.fit(x_train, y_train)
    return rf, np.array(scales).mean(), np.array(locs).mean(), np.array(means).mean(), np.array(stds).mean(), np.array(thresholds).mean()


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
    return create_model(x_train, y_train, params=rf_random.best_params_)


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
    model, threshold = create_model(x_train_sup, y_train, times_to_repeat=10)
    # model, threshold = model_selection(x_train_sup, y_train)

    print("Evaluating model...")
    y_pred = model.predict_proba(x_test_sup)
    y_pred = y_pred[:, 1]
    evaluation.evaluate(y_test, y_pred, threshold)
    # explainability.explain_dataset(model, x_train_sup, x_test_sup, threshold, y_test)