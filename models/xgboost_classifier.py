import sys
sys.path.append("/home/mpapale/thesis")
import xgboost as xgb
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from dataset_creation import sequences_crafting_for_classification, constants
from models import LSTM_classifier, evaluation, resampling_dataset, explainability
from scipy import stats

def model_selection(x_train, y_train):
    xg_reg = xgb.XGBClassifier()
    # first 3 parameters to directly control model complexity
    # last 2 parameters to add randomness to make training robust to noise.
    random_grid = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0, 0.5, 1, 1.5, 2, 5],
        'max_depth': [2, 3, 4, 5, 6, 8, 10],
        'learning_rate': [0.3, 0.2, 0.1, 0.01, 0.001],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.2, 0.4, 0.6, 0.8, 1.0]
    }

    xgb_random = GridSearchCV(estimator=xg_reg, param_grid=random_grid, cv=3, verbose=2, scoring="roc_auc",  n_jobs=-1)
    xgb_random.fit(x_train, y_train)

    print("Best params: ", xgb_random.best_params_)
    return create_model(x_train, y_train, params=xgb_random.best_params_)

def create_model(x_train, y_train, params=None, times_to_repeat=1):
    if not params:
        if constants.USING_AGGREGATED_FEATURES:
            if constants.DATASET_TYPE == constants.INJECTED_DATASET:
                params = constants.BEST_PARAMS_XGBOOST_AGGREGATED
            if constants.DATASET_TYPE == constants.REAL_DATASET:
                params = constants.BEST_PARAMS_XGBOOST_REAL_DATASET_AGGREGATED
            if constants.DATASET_TYPE == constants.OLD_DATASET:
                params = constants.BEST_PARAMS_XGBOOST_OLD_DATASET_AGGREGATED
        else:
            if constants.DATASET_TYPE == constants.INJECTED_DATASET:
                params = constants.BEST_PARAMS_XGBOOST_NO_AGGREGATED
            if constants.DATASET_TYPE == constants.REAL_DATASET:
                params = constants.BEST_PARAMS_XGBOOST_REAL_DATASET_NO_AGGREGATED
            if constants.DATASET_TYPE == constants.OLD_DATASET:
                params = constants.BEST_PARAMS_XGBOOST_OLD_DATASET_NO_AGGREGATED
    thresholds = []
    for i in range(times_to_repeat):
        print("iteration: ", i, "of", times_to_repeat)
        _x_val, _y_val, _x_train, _y_train = evaluation.get_val_test_set(x_train, y_train, 0.25)

        xg_reg = xgb.XGBClassifier(subsample=params["subsample"],
                                   min_child_weight=params["min_child_weight"],
                                   max_depth=params["max_depth"],
                                   learning_rate=params["learning_rate"],
                                   gamma=params["gamma"],
                                   colsample_bytree=params["colsample_bytree"])
        xg_reg.fit(_x_train, _y_train)
        y_pred = xg_reg.predict_proba(_x_val)[:, 1]
        thresholds.append(evaluation.find_best_threshold_fixed_fpr(_y_val, y_pred))
    print(np.array(thresholds).mean(), np.array(thresholds).std())
    xg_reg = xgb.XGBClassifier(subsample=params["subsample"],
                               min_child_weight=params["min_child_weight"],
                               max_depth=params["max_depth"],
                               learning_rate=params["learning_rate"],
                               gamma=params["gamma"],
                               colsample_bytree=params["colsample_bytree"])
    xg_reg.fit(x_train, y_train)
    return xg_reg, np.array(thresholds).mean()


def create_model_for_ensemble(x_train, y_train, params=None, times_to_repeat=1):
    def min_max_rescaling(values, min_, max_):
        return [(ith_element - min_) / (max_ - min_) for ith_element in values]
    if not params:
        if constants.USING_AGGREGATED_FEATURES:
            if constants.DATASET_TYPE == constants.INJECTED_DATASET:
                params = constants.BEST_PARAMS_XGBOOST_AGGREGATED
            if constants.DATASET_TYPE == constants.REAL_DATASET:
                params = constants.BEST_PARAMS_XGBOOST_REAL_DATASET_AGGREGATED
            if constants.DATASET_TYPE == constants.OLD_DATASET:
                params = constants.BEST_PARAMS_XGBOOST_OLD_DATASET_AGGREGATED
        else:
            if constants.DATASET_TYPE == constants.INJECTED_DATASET:
                params = constants.BEST_PARAMS_XGBOOST_NO_AGGREGATED
            if constants.DATASET_TYPE == constants.REAL_DATASET:
                params = constants.BEST_PARAMS_XGBOOST_REAL_DATASET_NO_AGGREGATED
            if constants.DATASET_TYPE == constants.OLD_DATASET:
                params = constants.BEST_PARAMS_XGBOOST_OLD_DATASET_NO_AGGREGATED

    avg_threshold = []
    avg_min = []
    avg_max = []
    avg_mean = []
    for i in range(times_to_repeat):
        print("iteration: ", i, "of", times_to_repeat)
        _x_val, _y_val, _x_train, _y_train = evaluation.get_val_test_set(x_train, y_train, 0.25)

        xg_reg = xgb.XGBClassifier(subsample=params["subsample"],
                                   min_child_weight=params["min_child_weight"],
                                   max_depth=params["max_depth"],
                                   learning_rate=params["learning_rate"],
                                   gamma=params["gamma"],
                                   colsample_bytree=params["colsample_bytree"])
        xg_reg.fit(_x_train, _y_train)
        y_pred = xg_reg.predict_proba(_x_val)[:, 1]
        min_ = min(y_pred)
        max_ = max(y_pred)
        mean_ = np.array(y_pred).mean()
        avg_min.append(min_)
        avg_max.append(max_)
        avg_mean.append(mean_)

        y_pred = min_max_rescaling(y_pred, min_, max_)
        avg_threshold.append(evaluation.find_best_threshold_fixed_fpr(_y_val, y_pred))

    xg_reg = xgb.XGBClassifier(subsample=params["subsample"],
                               min_child_weight=params["min_child_weight"],
                               max_depth=params["max_depth"],
                               learning_rate=params["learning_rate"],
                               gamma=params["gamma"],
                               colsample_bytree=params["colsample_bytree"])
    xg_reg.fit(x_train, y_train)
    return xg_reg, np.array(avg_threshold).mean(), np.array(avg_min).mean(), np.array(avg_max).mean(), np.array(avg_mean).mean()

def create_fit_model_for_ensemble_based_on_cdf(x_train, y_train, params=None, times_to_repeat=1):
    if not params:
        if constants.USING_AGGREGATED_FEATURES:
            if constants.DATASET_TYPE == constants.INJECTED_DATASET:
                params = constants.BEST_PARAMS_XGBOOST_AGGREGATED
            if constants.DATASET_TYPE == constants.REAL_DATASET:
                params = constants.BEST_PARAMS_XGBOOST_REAL_DATASET_AGGREGATED
            if constants.DATASET_TYPE == constants.OLD_DATASET:
                params = constants.BEST_PARAMS_XGBOOST_OLD_DATASET_AGGREGATED
        else:
            if constants.DATASET_TYPE == constants.INJECTED_DATASET:
                params = constants.BEST_PARAMS_XGBOOST_NO_AGGREGATED
            if constants.DATASET_TYPE == constants.REAL_DATASET:
                params = constants.BEST_PARAMS_XGBOOST_REAL_DATASET_NO_AGGREGATED
            if constants.DATASET_TYPE == constants.OLD_DATASET:
                params = constants.BEST_PARAMS_XGBOOST_OLD_DATASET_NO_AGGREGATED

    scales, locs, means, stds, thresholds = [], [], [], [], []
    for i in range(times_to_repeat):
        print("iteration: ", i, "of", times_to_repeat)

        _x_val, _y_val, _x_train, _y_train = evaluation.get_val_test_set(x_train, y_train, 0.25)
        model = xgb.XGBClassifier(subsample=params["subsample"],
                                   min_child_weight=params["min_child_weight"],
                                   max_depth=params["max_depth"],
                                   learning_rate=params["learning_rate"],
                                   gamma=params["gamma"],
                                   colsample_bytree=params["colsample_bytree"])
        model.fit(x_train, y_train)
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
    xg_reg = xgb.XGBClassifier(subsample=params["subsample"],
                               min_child_weight=params["min_child_weight"],
                               max_depth=params["max_depth"],
                               learning_rate=params["learning_rate"],
                               gamma=params["gamma"],
                               colsample_bytree=params["colsample_bytree"])
    xg_reg.fit(x_train, y_train)

    return xg_reg, np.array(scales).mean(), np.array(locs).mean(), np.array(means).mean(), np.array(stds).mean(), np.array(thresholds).mean()

if __name__ == "__main__":
    look_back = constants.LOOK_BACK

    x_train, y_train = sequences_crafting_for_classification.get_train_set()
    x_test, y_test = sequences_crafting_for_classification.get_test_set()

    # if the dataset is the real one -> contrast imbalanced dataset problem
    # if constants.DATASET_TYPE == constants.REAL_DATASET:
    #    x_train, y_train = resampling_dataset.oversample_set(x_train, y_train)

    # adapt train and test set to supervised learning without time windows
    x_train_sup = x_train[:, look_back, :]
    x_test_sup = x_test[:, look_back, :]

    print("Fitting model...")
    model, threshold = model_selection(x_train_sup, y_train)
    # model, threshold = create_model(x_train_sup, y_train, times_to_repeat=10)

    print("Evaluating model...")
    y_pred = model.predict_proba(x_test_sup)
    evaluation.evaluate(y_test, y_pred[:, 1], threshold)

    # sample = np.reshape(x_test_sup[1], (1, x_test_sup.shape[1]))
    # explainability.explain_dataset(model, x_train_sup, sample, threshold)
    # explainability.explain_dataset(model, x_train_sup, x_test_sup, threshold, y_test)