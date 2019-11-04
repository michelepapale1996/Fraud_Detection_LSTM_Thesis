import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
# seaborn can generate several warnings, we ignore them
import warnings
from sklearn.model_selection import train_test_split
from scipy import stats
from models import LSTM
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
from sklearn.model_selection import RandomizedSearchCV
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.optimizers import Adam, RMSprop
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
import random

# in order to print all the columns
pd.set_option('display.max_columns', 100)
sns.set(style="white", color_codes=True)
sns.set_context(rc={"font.family":'sans',"font.size":24,"axes.titlesize":24,"axes.labelsize":24})
warnings.filterwarnings("ignore")
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# -----------------------------------------------------------------------------------------------
# ---------Main idea: creating an LSTM predictor for the next transaction of a user--------------
# -------The LSTM network is trained using sequences belonging to same user for all users--------
# -----------------------------------------------------------------------------------------------

look_back = 24
num_features = 9

x = np.load("/home/mpapale/thesis/anomaly_detection/dataset_" + str(look_back) + "_lookback/x.npy")
y = np.load("/home/mpapale/thesis/anomaly_detection/dataset_" + str(look_back) + "_lookback/y.npy")

x = x[:, 8:24, :]
look_back = 15

print("Lookback: ", look_back)
print("x shape: ", x.shape, "y shape: ", y.shape, " before dataset engineering.")

x_no_frauds = x[np.where(y[:,9] == 0)]
y_no_frauds = y[np.where(y[:,9] == 0)]
x_frauds = x[np.where(y[:,9] == 1)]
y_frauds = y[np.where(y[:,9] == 1)]

x_train_no_frauds, x_test_no_frauds, y_train_no_frauds, y_test_no_frauds = train_test_split(x_no_frauds, y_no_frauds, test_size=0.33, random_state=seed)
x_train_no_frauds, x_val_no_frauds, y_train_no_frauds, y_val_no_frauds = train_test_split(x_train_no_frauds, y_train_no_frauds, test_size=0.33, random_state=seed)
x_val_frauds, x_test_frauds, y_val_frauds, y_test_frauds = train_test_split(x_frauds, y_frauds, test_size=0.6, random_state=seed)

# downsample x to have (constant * num_frauds) sequences of transactions
# returns x_downsampled, y_dawnsampled, x_not_used, y_not_used
def downsample(x, y, num_frauds):
    x_resampled = np.zeros((0, look_back + 1, num_features))
    y_resampled = np.zeros((0, num_features + 1))
    for _ in range(num_frauds * 100):
        position = random.randint(0, len(x) - 1)

        x_picked = np.reshape(x[position], (1, look_back + 1, num_features))
        y_picked = np.reshape(y[position], (1, num_features + 1))

        x = np.delete(x, position, axis=0)
        y = np.delete(y, position, axis=0)

        x_resampled = np.concatenate((x_resampled, x_picked))
        y_resampled = np.concatenate((y_resampled, y_picked))
    return x_resampled, y_resampled, x, y


x_val_no_frauds, y_val_no_frauds, x_val_not_used, y_val_not_used = downsample(x_val_no_frauds, y_val_no_frauds, len(y_val_frauds))
x_val = np.concatenate([x_val_frauds, x_val_no_frauds])
y_val = np.concatenate([y_val_frauds, y_val_no_frauds])

x_test_no_frauds, y_test_no_frauds, x_test_not_used, y_test_not_used = downsample(x_test_no_frauds, y_test_no_frauds, len(y_test_frauds))
x_test = np.concatenate([x_test_frauds, x_test_no_frauds])
y_test = np.concatenate([y_test_frauds, y_test_no_frauds])

x_train_no_frauds = np.concatenate([x_train_no_frauds, x_val_not_used, x_test_not_used])
y_train_no_frauds = np.concatenate([y_train_no_frauds, y_val_not_used, y_test_not_used])

# splitting train_no_frauds set in train and val (validation used to find the mean and cov of the multivariate normal distr)
x_train_no_frauds, x_val_no_frauds, y_train_no_frauds, y_val_no_frauds = train_test_split(x_train_no_frauds, y_train_no_frauds, test_size=0.5, random_state=seed)

np.random.shuffle(x_train_no_frauds)
np.random.shuffle(y_train_no_frauds)
#np.random.shuffle(x_val_no_frauds)
#np.random.shuffle(y_val_no_frauds)
np.random.shuffle(x_val)
np.random.shuffle(y_val)
np.random.shuffle(x_test)
np.random.shuffle(y_test)

print("In training there are ", len(y_train_no_frauds), " sequences.")
print("In validation (to find the mean and cov) there are ", len(y_val_no_frauds), " sequences.")
print("In validation there are ", len(y_val_frauds), " frauds and ", len(y_val_no_frauds), " no frauds.")
print("In testing there are ", len(y_test_frauds), " frauds and ", len(y_test_no_frauds), " no frauds.")

# delete from y the "isFraud" label
y_train_no_frauds = np.delete(y_train_no_frauds, 9, axis=1)

# x_train_no_frauds, y_train_no_frauds will be used to train the model
# x_test_no_frauds, y_test_no_frauds, x_test_frauds, y_test_frauds will be used to evaluate the model
# x_val_no_frauds, y_val_no_frauds, x_val_frauds, y_val_frauds will be used to find the decision boundary

# define the grid search parameters
batch_size = [1, 5]
epochs = [10]
layers = [{'input': 10, 'output': num_features}]#,
          #{'input': 128, 'output': num_features},
          #{'input': 128, 'hidden1': 64, 'output': num_features},
          #{'input': 256, 'hidden1': 128, 'output': num_features},
          #{'input': 512, 'hidden1': 256, 'hidden2': 128, 'output': num_features}]
learning_rate = [0.1, 0.01, 0.001]
dropout_rate = [0.0, 0.3, 0.8]

def create_model(layers={'input': 32, 'output': 9}, learning_rate=0.0001, dropout_rate=0.3, look_back=15):
    model = Sequential()
    n_hidden = len(layers) - 2
    if n_hidden > 2:
        model.add(LSTM(layers['input'], input_shape=(look_back+1, num_features), return_sequences=True))

        # add hidden layers return sequence true
        for i in range(2, n_hidden):
            model.add(LSTM(layers["hidden" + str(i)], return_sequences=True))
            model.add(Dropout(dropout_rate))
        # add hidden_last return Sequences False
        model.add(LSTM(layers['hidden' + str(n_hidden)], return_sequences=False))
        model.add(Dropout(dropout_rate))
    else:
        model.add(LSTM(layers['input'], input_shape=(look_back+1, num_features), return_sequences=False))

    model.add(Dense(layers['output'], activation='sigmoid'))

    model.compile(loss="mae", optimizer=Adam(lr=learning_rate))
    # model.summary()
    return model

# This function adjusts class predictions based on the prediction threshold (t).
def adjusted_classes(y_scores, t):
    return [1 if y >= t else 0 for y in y_scores]

def get_errors(y_true, y_pred):
    len_y = len(y_pred)
    errors = []
    for i in range(len_y):
        error = mean_absolute_error(y_true[i, 0:9], y_pred[i])
        errors.append(error)
    return errors

def get_probabilities(y, y_pred, distribution):
    return distribution.pdf(np.absolute(y[:, 0:9] - y_pred))

def mean_and_covariance(y, y_pred):
    errors = np.absolute(y[:, 0:9] - y_pred)
    return np.mean(errors, axis=0), np.std(errors, axis=0)

def find_best_threshold(y_true, y_pred, distribution):
    # probabilities = get_probabilities(y_true, y_pred, distribution)
    probabilities = get_errors(y_true, y_pred)
    # thresholds = np.linspace(probabilities.min(), probabilities.max(), num=200, endpoint=True)
    thresholds = np.linspace(min(probabilities), max(probabilities), num=200, endpoint=True)
    scores = {}
    for t in thresholds:
        labels = adjusted_classes(probabilities, t)
        f = fbeta_score(y_true[:,9], labels, beta=0.1)
        scores[t] = f
    return max(scores, key=scores.get)

def evaluate(y_true, y_pred, threshold):
    print("---- Performance ----")
    print("Roc_auc score: ", roc_auc_score(y_true, y_pred))
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    precision_recall_auc = auc(recall, precision)
    print("Precision_recall_auc score: %.3f" % precision_recall_auc)
    print("Average precision: ", average_precision_score(y_true, y_pred))
    y_pred = adjusted_classes(y_pred, threshold)
    print("Precision: ", precision_score(y_true, y_pred))
    print("Recall: ", recall_score(y_true, y_pred))
    print("f1 score: ", f1_score(y_true, y_pred))
    print("f0.1 score: ", fbeta_score(y_true, y_pred, beta=0.1))
    print("Average accuracy: ", balanced_accuracy_score(y_true, y_pred))
    print("accuracy: ", accuracy_score(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))


model = create_model()
# fit model
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
model.fit(x_train_no_frauds, y_train_no_frauds, epochs=3, validation_split=0.1, verbose=1, callbacks=[es])

'''
# getting the mean and cov of the multivariate normal distribution
y_val_pred_no_frauds = model.predict(x_val_no_frauds)
mean, cov = mean_and_covariance(y_val_no_frauds, y_val_pred_no_frauds)
distribution = stats.multivariate_normal(mean=mean, cov=cov)
'''

# using validation set to find the best threshold to divide frauds and genuine
y_val_pred = model.predict(x_val)
#                best_threshold = find_best_threshold(y_val, y_val_pred, distribution)
best_threshold = find_best_threshold(y_val, y_val_pred, False)

# calculating the model performance
y_test_pred = model.predict(x_test)
# errors = get_errors(y_test, y_test_pred)
#                probabilities = get_probabilities(y_test, y_test_pred, distribution)
probabilities = get_errors(y_test, y_test_pred)
labels = adjusted_classes(probabilities, best_threshold)
evaluate(y_test[:,9], labels, best_threshold)

'''
model = KerasRegressor(build_fn=create_model, verbose=0)
param_grid = dict(batch_size=batch_size,
                  epochs=epochs,
                  layers=layers,
                  learning_rate=learning_rate,
                  dropout_rate=dropout_rate)
# scoring = make_scorer(evaluate_performance, needs_threshold=True)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid = RandomizedSearchCV(estimator=model,
                          param_distributions=param_grid,
                          n_jobs=-1,
                          cv=3,
                          n_iter=2,
                          verbose=2)
                          #scoring=scoring)
grid_result = grid.fit(x_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

print("Best model score on test set", grid_result.score(x_test, y_test))
'''