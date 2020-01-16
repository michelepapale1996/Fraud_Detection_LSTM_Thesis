import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from dataset_creation import sequences_crafting_for_classification, constants
from models import LSTM_classifier, evaluation
from sklearn.model_selection import RandomizedSearchCV
import warnings
sns.set(style="white", color_codes=True)
sns.set_context(rc={"font.family":'sans',"font.size":24,"axes.titlesize":24,"axes.labelsize":24})
# seaborn can generate several warnings, we ignore them
warnings.filterwarnings("ignore")


# in order to print all the columns
pd.set_option('display.max_columns', 100)


def create_model(x_train, y_train):
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    return rf

def model_selection(x_train, y_train):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=100, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
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
                                   n_iter=100,
                                   cv=3,
                                   verbose=2,
                                   random_state=42,
                                   n_jobs=-1)

    # Fit the random search model
    rf_random.fit(x_train, y_train)

    print("Best params: ", rf_random.best_params_)
    best_random = rf_random.best_estimator_
    return best_random


look_back = constants.LOOK_BACK

x_train, y_train = sequences_crafting_for_classification.create_train_set(look_back)
x_test, y_test = sequences_crafting_for_classification.create_test_set(look_back)
# adapt train and test set to supervised learning without time windows
x_train_sup = x_train[:, look_back, :]
x_test_sup = x_test[:, look_back, :]


print("Fitting model...")
model = create_model(x_train_sup, y_train)
# model = model_selection(x_train_sup, y_train)

print("Evaluating model...")
y_pred = model.predict_proba(x_test_sup)
evaluation.evaluate_n_times(y_test, y_pred[:, 1])