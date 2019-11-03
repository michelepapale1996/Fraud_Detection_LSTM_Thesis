import pandas as pd
import numpy as np
import random
from sklearn import preprocessing
import seaborn as sns
# seaborn can generate several warnings, we ignore them
import warnings
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, roc_curve, roc_auc_score, precision_recall_curve, auc, make_scorer, confusion_matrix, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.optimizers import Adam, RMSprop
from keras.models import Sequential
from imblearn.under_sampling import RandomUnderSampler
from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split
# in order to print all the columns
pd.set_option('display.max_columns', 100)
sns.set(style="white", color_codes=True)
sns.set_context(rc={"font.family": 'sans', "font.size": 24, "axes.titlesize": 24, "axes.labelsize": 24})
warnings.filterwarnings("ignore")

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# -----------------------------------------------------------------------------------------------
# --------------------------Main idea: creating an LSTM classifier ------------------------------
# -----------------------------------------------------------------------------------------------

look_back = 9
num_features = 9

def undersample(x, y):
    num_frauds = np.unique(y, return_counts=True)[1][1]
    x_no_frauds = x[np.where(y == 0)]
    x_resampled = np.zeros((0, look_back + 1, num_features))
    y_resampled = np.zeros(0)
    for _ in range(num_frauds * 10):
        reshaped = np.reshape(x_no_frauds[random.randint(0, len(x_no_frauds) - 1)], (1, look_back + 1, num_features))
        x_resampled = np.concatenate((x_resampled, reshaped))
        y_resampled = np.append(y_resampled, 0)
    x_resampled = np.concatenate((x_resampled, x[np.where(y == 1)]))
    y_resampled = np.append(y_resampled, [1 for _ in range(num_frauds)])
    np.random.shuffle(x_resampled)
    np.random.shuffle(y_resampled)
    return x_resampled, y_resampled


x = np.load("/home/mpapale/thesis/classification/dataset_" + str(look_back) + "_lookback/x.npy")
y = np.load("/home/mpapale/thesis/classification/dataset_" + str(look_back) + "_lookback/y.npy")
print("Il ratio frodi vs genuine è: ")
unique, counts = np.unique(y, return_counts=True)
print(counts)
x, y = undersample(x, y)
print("Dopo l'undersampling il ratio frodi vs genuine è: ")
unique, counts = np.unique(y, return_counts=True)
print(counts)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
print("train_labels")
print(np.unique(y_train, return_counts=True))
print("test_labels")
print(np.unique(y_test, return_counts=True))

# define the grid search parameters
batch_size = [1, 5]
epochs = [10]
layers = [{'input': 10, 'output': 1}]#,
          #{'input': 128, 'output': 1},
          #{'input': 128, 'hidden1': 64, 'output': 1},
          #{'input': 256, 'hidden1': 128, 'output': 1},
          #{'input': 512, 'hidden1': 256, 'hidden2': 128, 'output': 1}]
learning_rate = [0.1, 0.01, 0.001]
dropout_rate = [0.0, 0.3, 0.8]


def create_model(layers={'input': 5, 'output': 1}, learning_rate=0.01, dropout_rate=0, look_back=25):
    model = Sequential()
    n_hidden = len(layers) - 2
    if n_hidden > 2:
        model.add(LSTM(layers['input'], input_shape=(look_back + 1, num_features), return_sequences=True))

        # add hidden layers return sequence true
        for i in range(2, n_hidden):
            model.add(LSTM(layers["hidden" + str(i)], return_sequences=True))
            model.add(Dropout(dropout_rate))
        # add hidden_last return Sequences False
        model.add(LSTM(layers['hidden' + str(n_hidden)], return_sequences=False))
        model.add(Dropout(dropout_rate))
    else:
        model.add(LSTM(layers['input'], input_shape=(look_back + 1, num_features), return_sequences=False))

    # add output
    model.add(Dense(layers['output'], activation='sigmoid'))

    # compile model and print summary
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=learning_rate))
    # model.summary()
    return model


def evaluate_performance(y_true, y_pred):
    print("---- Performance ----")
    roc_auc = roc_auc_score(y_true, y_pred)
    print("Roc_auc score: %.3f" % roc_auc)

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    def get_position(recall):
        for i in range(0, len(recall)):
            if recall[i] < 0.2:
                return i - 1

    # This function adjusts class predictions based on the prediction threshold (t).
    def adjusted_classes(y_scores, t):
        return [1 if y >= t else 0 for y in y_scores]

    precision_recall_auc = auc(recall, precision)
    print("Precision_recall_auc score: %.3f" % precision_recall_auc)
    position = get_position(recall)
    print("Precision: " + str(precision[position]))
    threshold = thresholds[position]
    y_pred = adjusted_classes(y_pred, threshold)
    f1 = f1_score(y_true, y_pred)
    print("f1 score: %.3f" % f1)
    print("accuracy: " + str(accuracy_score(y_true, y_pred)))
    print(confusion_matrix(y_true, y_pred))
    return f1


model = KerasClassifier(build_fn=create_model, verbose=0)
param_grid = dict(batch_size=batch_size,
                  epochs=epochs,
                  layers=layers,
                  learning_rate=learning_rate,
                  dropout_rate=dropout_rate)
scoring = make_scorer(evaluate_performance, needs_threshold=True)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid = RandomizedSearchCV(estimator=model,
                          param_distributions=param_grid,
                          n_jobs=-1,
                          cv=3,
                          n_iter=2,
                          verbose=2,
                          scoring=scoring)
grid_result = grid.fit(x_train, y_train)
# summarize results
print("Best f1: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

print("Best model score on test set")
grid_result.score(x_test, y_test)