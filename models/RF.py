import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.append("/home/mpapale/thesis")
from dataset_creation import sequences_crafting_for_classification, constants
from dataset_creation.constants import BEST_PARAMS_RF
from models import LSTM_classifier, evaluation
from sklearn.model_selection import RandomizedSearchCV
import lime
import lime.lime_tabular
from lime import submodular_pick


def create_model(x_train, y_train, params=None):
    if not params:
        if constants.DATASET_TYPE == constants.INJECTED_DATASET:
            params = BEST_PARAMS_RF
        if constants.DATASET_TYPE == constants.REAL_DATASET:
            params = constants.BEST_PARAMS_RF_REAL_DATASET
        if constants.DATASET_TYPE == constants.OLD_DATASET:
            params = constants.BEST_PARAMS_RF_OLD_DATASET

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
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10, 20]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4, 8, 10]
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
    rf_random = RandomizedSearchCV(estimator=rf,
                                   param_distributions=random_grid,
                                   n_iter=500,
                                   cv=3,
                                   scoring="roc_auc",
                                   verbose=1,
                                   random_state=42,
                                   n_jobs=-1)

    # Fit the random search model
    rf_random.fit(x_train, y_train)

    print("Best params: ", rf_random.best_params_)
    best_random = rf_random.best_estimator_
    return best_random


if __name__ == "__main__":
    look_back = constants.LOOK_BACK

    x_train, y_train = sequences_crafting_for_classification.get_train_set()
    x_test, y_test = sequences_crafting_for_classification.get_test_set()
    # adapt train and test set to supervised learning without time windows
    x_train_sup = x_train[:, look_back, :]
    x_test_sup = x_test[:, look_back, :]


    print("Fitting model...")
    model = create_model(x_train_sup, y_train)
    # model = model_selection(x_train_sup, y_train)

    print("Evaluating model...")
    y_pred = model.predict_proba(x_test_sup)
    evaluation.evaluate_n_times(y_test, y_pred[:, 1])


    explainer = lime.lime_tabular.LimeTabularExplainer(x_train_sup, feature_names=['Importo', 'MsgErrore', 'NumConfermaSMS', 'isItalianSender', 'isItalianReceiver', 'time_delta'], class_names=["Genuine", "Fraud"], discretize_continuous=True)
    i = np.random.randint(0, x_test_sup.shape[0])
    exp = explainer.explain_instance(x_test_sup[i], model.predict_proba, num_features=6, top_labels=6)
    # exp.save_to_file("explanation.html")

    sp_obj = submodular_pick.SubmodularPick(explainer, x_test_sup, model.predict_proba, num_features=6, num_exps_desired=10)

    i = 0
    for exp in sp_obj.sp_explanations:
        exp.save_to_file("explanation_" + str(i) + ".html")
        i += 1