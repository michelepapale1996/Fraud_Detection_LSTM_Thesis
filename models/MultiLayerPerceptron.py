import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from dataset_creation import sequences_crafting_for_classification, constants
from models import LSTM_classifier, evaluation
from sklearn.model_selection import RandomizedSearchCV
import warnings
from keras.wrappers import scikit_learn
from adversarial_attacks import fgsm
sns.set(style="white", color_codes=True)
sns.set_context(rc={"font.family":'sans',"font.size":24,"axes.titlesize":24,"axes.labelsize":24})
# seaborn can generate several warnings, we ignore them
warnings.filterwarnings("ignore")

def create_fit_model(x_train, y_train):
    model = Sequential()
    model.add(Dense(64, input_dim=len(x_train[0]), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.fit(x_train, y_train, epochs=30, verbose=1, batch_size=30)

    return model

def create_model(layers, dropout_rate, look_back, num_features):
    model = Sequential()
    n_hidden = len(layers) - 2
    if n_hidden > 2:
        model.add(Dense(layers['input'], input_dim=num_features))

        # add hidden layers return sequence true
        for i in range(2, n_hidden):
            model.add(Dense(layers["hidden" + str(i)]))
            model.add(Dropout(dropout_rate))
        # add hidden_last return Sequences False
        model.add(Dense(layers['hidden' + str(n_hidden)]))
        model.add(Dropout(dropout_rate))
    else:
        model.add(Dense(layers['input'], input_dim=num_features))

    model.add(Dense(layers['output'], activation='sigmoid'))
    model.compile(loss="mse", optimizer="adam")
    # model.summary()
    return model

def model_selection(x, y):
    # define the grid search parameters
    batch_size = [8, 16, 32, 64, 128, 256]
    epochs = [2, 5, 10, 25, 50, 100, 500]
    layers = [{'input': 32, 'hidden1': 32, 'output': 1},
              {'input': 64, 'hidden1': 64, 'output': 1},
              {'input': 96, 'hidden1': 32, 'output': 1},
              {'input': 128, 'hidden1': 64, 'hidden2': 32, 'output': 1},
              {'input': 256, 'hidden1': 64, 'hidden2': 256, 'output': 1},
              {'input': 128, 'hidden1': 64, 'hidden2': 64, 'hidden3': 128,'output': 1},
              {'input': 512, 'hidden1': 256, 'hidden2': 256, 'hidden3': 512,'output': 1}]
    dropout_rate = [0.2, 0.3, 0.5, 0.6, 0.8]

    # batch_size = np.random.choice(batch_size, int(len(batch_size) / 2))
    # epochs = np.random.choice(epochs, int(len(epochs) / 2))
    # layers = np.random.choice(layers, int(len(layers) / 2))
    # learning_rate = np.random.choice(learning_rate, int(len(learning_rate) / 2))
    # dropout_rate = np.random.choice(dropout_rate, int(len(dropout_rate) / 2))

    model = scikit_learn.KerasClassifier(build_fn=create_model, look_back=look_back, num_features=len(x[0, :]), verbose=1)
    param_grid = dict(layers=layers, batch_size=batch_size, epochs=epochs, dropout_rate=dropout_rate)

    '''
    validation_fold = [-1 for _ in range(len(x_train))] + [0 for _ in range(len(x_val))]
    x_train = np.append(x_train, x_val, axis=0)
    y_train = np.append(y_train, y_val, axis=0)
    ps = PredefinedSplit(validation_fold)
    grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100, n_jobs=-1, cv=ps, scoring="roc_auc", verbose=1)
    '''

    grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=1000, n_jobs=-1, cv=3, scoring="roc_auc", verbose=1)

    grid_result = grid.fit(x, y)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    return grid_result

if __name__ == "__main__":
    look_back = constants.LOOK_BACK
    x_train, y_train = sequences_crafting_for_classification.get_train_set()
    x_test, y_test = sequences_crafting_for_classification.get_test_set()

    # adapt train and test set to supervised learning without time windows
    x_train_sup = x_train[:, look_back, :]
    x_test_sup = x_test[:, look_back, :]

    # print("Fitting model...")
    # model = create_fit_model(x_train_sup, y_train)

    print("Model selection...")
    model = model_selection(x_train_sup, y_train)

    print("Evaluating model...")
    y_pred = model.predict_proba(x_test_sup)
    evaluation.evaluate(y_test, y_pred.ravel(), threshold)
