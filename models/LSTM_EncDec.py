import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
# seaborn can generate several warnings, we ignore them
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from models import LSTM
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.layers import TimeDistributed
from keras.optimizers import Adam, RMSprop
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor

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

print("Input shape:")
print(x.shape)
x = x[0:10000]
y = y[0:10000]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

print("Train input shape:")
print(x_train.shape)
print("Test input shape:")
print(x_test.shape)

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

def create_model(layers={'input': 128, 'output': 9}, learning_rate=0.01, dropout_rate=0.3, look_back=24):
    model = Sequential()
    model.add(LSTM(128, activation='relu', input_shape=(look_back + 1, num_features), return_sequences=True))
    model.add(LSTM(64, activation='relu', return_sequences=False))
    model.add(RepeatVector(look_back + 1))
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(num_features)))
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # compile model and print summary
    # model.compile(loss="binary_crossentropy", optimizer=Adam(lr=learning_rate))
    losses = ["mse",
              "binary_crossentropy",
              "binary_crossentropy",
              "binary_crossentropy",
              "mse",
              "binary_crossentropy",
              "binary_crossentropy",
              "binary_crossentropy",
              "binary_crossentropy",]
    model.compile(loss="mse", optimizer=Adam(lr=learning_rate))
    # model.summary()
    return model

def to_binary(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)

    def get_position(recall):
        for i in range(0, len(recall)):
            if recall[i] < 0.2:
                return i - 1

    # This function adjusts class predictions based on the prediction threshold (t).
    def adjusted_classes(y_scores, t):
        return [1 if y >= t else 0 for y in y_scores]

    threshold = thresholds[get_position(recall)]
    y_pred = adjusted_classes(y_pred, threshold)
    return y_pred


model = create_model()
# fit model
model.fit(x_train, x_train, epochs=20, verbose=1)
y_pred = model.predict(x_test, verbose=1)
y_pred[:,1] = to_binary(y_test[:,1], y_pred[:,1])
y_pred[:,2] = to_binary(y_test[:,2], y_pred[:,2])
y_pred[:,3] = to_binary(y_test[:,3], y_pred[:,3])
y_pred[:,5] = to_binary(y_test[:,5], y_pred[:,5])
y_pred[:,6] = to_binary(y_test[:,6], y_pred[:,6])
y_pred[:,7] = to_binary(y_test[:,7], y_pred[:,7])
y_pred[:,8] = to_binary(y_test[:,8], y_pred[:,8])
print("Predicted: ", y_pred)
print("True: ", y_test)
print("How good is: ", model.evaluate(x_test, y_test))
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