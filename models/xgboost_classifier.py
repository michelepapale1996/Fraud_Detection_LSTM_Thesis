import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from dataset_creation import sequences_crafting_for_classification, constants
from models import LSTM_classifier, evaluation
from dataset_creation.constants import BEST_PARAMS_XGBOOST

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

    xgb_random = RandomizedSearchCV(estimator=xg_reg,
                                    param_distributions=random_grid,
                                    n_iter=500,
                                    cv=3,
                                    verbose=2,
                                    scoring="roc_auc",
                                    random_state=42,
                                    n_jobs=-1)
    xgb_random.fit(x_train, y_train)

    print("Best params: ", xgb_random.best_params_)
    best_random = xgb_random.best_estimator_
    return best_random

def create_model(x_train, y_train, params=None):
    if not params:
        if constants.DATASET_TYPE == constants.INJECTED_DATASET:
            params = BEST_PARAMS_XGBOOST
        if constants.DATASET_TYPE == constants.REAL_DATASET:
            params = constants.BEST_PARAMS_XGBOOST_REAL_DATASET
        if constants.DATASET_TYPE == constants.OLD_DATASET:
            params = constants.BEST_PARAMS_XGBOOST_OLD_DATASET

    xg_reg = xgb.XGBClassifier(subsample=params["subsample"],
                               min_child_weight=params["min_child_weight"],
                               max_depth=params["max_depth"],
                               learning_rate=params["learning_rate"],
                               gamma=params["gamma"],
                               colsample_bytree=params["colsample_bytree"])
    xg_reg.fit(x_train, y_train)
    return xg_reg


if __name__ == "__main__":
    look_back = constants.LOOK_BACK

    x_train, y_train = sequences_crafting_for_classification.get_train_set()
    x_test, y_test = sequences_crafting_for_classification.get_test_set()

    # adapt train and test set to supervised learning without time windows
    x_train_sup = x_train[:, look_back, :]
    x_test_sup = x_test[:, look_back, :]

    print("Fitting model...")
    # model = model_selection(x_train_sup, y_train)
    model = create_model(x_train_sup, y_train)

    print("Evaluating model...")
    y_pred = model.predict_proba(x_test_sup)
    evaluation.evaluate_n_times(y_test, y_pred[:, 1])