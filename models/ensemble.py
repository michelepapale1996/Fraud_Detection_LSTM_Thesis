from models import LSTM_classifier, evaluation
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from dataset_creation import sequences_crafting_for_classification
from dataset_creation.constants import *
from adversarial_attacks import fgsm
from models import MultiLayerPerceptron, xgboost_classifier, RF
from scipy import stats

def get_predictions_for_each_model(lstm, rf, xg_reg, x, x_supervised):
    y_lstm = lstm.predict(x)
    y_rf = rf.predict_proba(x_supervised)
    y_xgb = xg_reg.predict_proba(x_supervised)
    return y_lstm.ravel(), y_rf[:, 1], y_xgb[:,1]

def mean_and_covariance(y, y_pred):
    errors = np.absolute(y - y_pred)
    return np.mean(errors, axis=0), np.cov(errors, axis=0)

# --------------------------- start of the types of ensembles ----------------------------
# sum of the models outputs on validation set
# find the best threshold
# sum of the models outputs on test set
# adjust output based on threshold
def predict_test_based_on_sum(lstm, rf, xg_reg, x_val, x_val_supervised, y_val, x_test, x_test_supervised):
    y_val_pred_lstm, y_val_pred_rf, y_val_pred_xgb = get_predictions_for_each_model(lstm, rf, xg_reg, x_val, x_val_supervised)
    len_ = len(y_val_pred_lstm)

    def min_max_rescaling(values):
        min_ = min(values)
        max_ = max(values)
        return [(ith_element - min_) / (max_ - min_) for ith_element in values]

    y_val_pred_lstm = min_max_rescaling(y_val_pred_lstm)
    y_val_pred_rf = min_max_rescaling(y_val_pred_rf)
    y_val_pred_xgb = min_max_rescaling(y_val_pred_xgb)

    y_val_pred = y_val_pred_lstm * np.array([0.333] * len_) + y_val_pred_rf * np.array([0.333] * len_) + y_val_pred_xgb * np.array([0.333] * len_)

    threshold = evaluation.find_best_threshold_fixed_fpr(y_val, y_val_pred)

    y_pred_lstm, y_pred_rf, y_pred_xgb = get_predictions_for_each_model(lstm, rf, xg_reg, x_test, x_test_supervised)
    len_ = len(y_pred_lstm)
    y_pred_lstm = min_max_rescaling(y_pred_lstm)
    y_pred_rf = min_max_rescaling(y_pred_rf)
    y_pred_xgb = min_max_rescaling(y_pred_xgb)
    y_pred = y_pred_lstm * np.array([0.333] * len_) + y_pred_rf * np.array([0.333] * len_) + y_pred_xgb * np.array([0.333] * len_)

    y_test_pred = evaluation.adjusted_classes(y_pred, threshold)
    return y_test_pred

# each model gives his opinion and the output is based on the majority
def predict_test_based_on_voting(lstm, rf, xg_reg, x_val, x_val_supervised, y_val, x_test, x_test_supervised, y_test):
    y_val_pred_lstm, y_val_pred_rf, y_val_pred_xgb = get_predictions_for_each_model(lstm, rf, xg_reg, x_val, x_val_supervised)

    # get threshold for each model
    threshold_lstm = evaluation.find_best_threshold_fixed_fpr(y_val, y_val_pred_lstm)
    threshold_rf = evaluation.find_best_threshold_fixed_fpr(y_val, y_val_pred_rf)
    threshold_xgb = evaluation.find_best_threshold_fixed_fpr(y_val, y_val_pred_xgb)

    # predicting test set
    y_pred_lstm, y_pred_rf, y_pred_xgb = get_predictions_for_each_model(lstm, rf, xg_reg, x_test, x_test_supervised)

    y_pred_lstm = np.array(evaluation.adjusted_classes(y_pred_lstm, threshold_lstm))
    y_pred_rf = np.array(evaluation.adjusted_classes(y_pred_rf, threshold_rf))
    y_pred_xgb = np.array(evaluation.adjusted_classes(y_pred_xgb, threshold_xgb))

    y_test_pred = []

    lstm_indices_found = evaluation.get_indices(y_test, y_pred_lstm)
    rf_indices_found = evaluation.get_indices(y_test, y_pred_rf)
    xgboost_indices_found = evaluation.get_indices(y_test, y_pred_xgb)

    not_found_by_xgboost, not_found_by_rf, not_found_by_others = evaluation.get_jaccard_index(lstm_indices_found, rf_indices_found, xgboost_indices_found)

    for i in range(len(y_pred_lstm)):
        if y_pred_lstm[i] + y_pred_rf[i] + y_pred_xgb[i] >= 2:
            y_test_pred.append(1)
        else:
            y_test_pred.append(0)

    return y_test_pred, not_found_by_xgboost, not_found_by_rf, not_found_by_others

# for each model, get threshold and get mean of the output on validation set .
# on test set, calculate the distance from the calculated mean.
# The model that has the max distance will be the one who decides
def predict_test_based_on_more_confident(lstm, rf, xg_reg, x_val, x_val_supervised, y_val, x_test, x_test_supervised, y_test):
    y_val_pred_lstm, y_val_pred_rf, y_val_pred_xgb = get_predictions_for_each_model(lstm, rf, xg_reg, x_val, x_val_supervised)

    def min_max_rescaling(values, min_, max_):
        return [(ith_element - min_) / (max_ - min_) for ith_element in values]

    lstm_min = min(y_val_pred_lstm)
    lstm_max = max(y_val_pred_lstm)
    rf_min = min(y_val_pred_rf)
    rf_max = max(y_val_pred_rf)
    xgb_min = min(y_val_pred_xgb)
    xgb_max = max(y_val_pred_xgb)
    y_val_pred_lstm = min_max_rescaling(y_val_pred_lstm, lstm_min, lstm_max)
    y_val_pred_rf = min_max_rescaling(y_val_pred_rf, rf_min, rf_max)
    y_val_pred_xgb = min_max_rescaling(y_val_pred_xgb, xgb_min, xgb_max)

    # get threshold for each model
    threshold_lstm = evaluation.find_best_threshold_fixed_fpr(y_val, y_val_pred_lstm)
    threshold_rf = evaluation.find_best_threshold_fixed_fpr(y_val, y_val_pred_rf)
    threshold_xgb = evaluation.find_best_threshold_fixed_fpr(y_val, y_val_pred_xgb)

    # get mean for each model
    lstm_mean = np.array(y_val_pred_lstm).mean()
    rf_mean = np.array(y_val_pred_rf).mean()
    xgb_mean = np.array(y_val_pred_xgb).mean()

    # get std for each model
    lstm_std = np.array(y_val_pred_lstm).std()
    rf_std = np.array(y_val_pred_rf).std()
    xgb_std = np.array(y_val_pred_xgb).std()

    # predicting test set
    y_pred_lstm, y_pred_rf, y_pred_xgb = get_predictions_for_each_model(lstm, rf, xg_reg, x_test, x_test_supervised)
    y_pred_lstm = min_max_rescaling(y_pred_lstm, lstm_min, lstm_max)
    y_pred_rf = min_max_rescaling(y_pred_rf, rf_min, rf_max)
    y_pred_xgb = min_max_rescaling(y_pred_xgb, xgb_min, xgb_max)

    y_test_pred = []
    num_decisions_taken_by_lstm = 0
    num_decisions_correctly_taken_from_lstm = 0
    num_decisions_correctly_taken_from_lstm_and_not_from_xgb_or_rf = 0
    num_decisions_taken_by_rf = 0
    num_decisions_taken_by_xgb = 0
    for i in range(len(y_pred_lstm)):
        max_value = max(abs(y_pred_lstm[i] - lstm_mean), abs(y_pred_rf[i] - rf_mean), abs(y_pred_xgb[i] - xgb_mean))
        if abs(y_pred_lstm[i] - lstm_mean) == max_value:
            num_decisions_taken_by_lstm += 1
            lstm_output = 1 if y_pred_lstm[i] > threshold_lstm else 0
            rf_output = 1 if y_pred_rf[i] > threshold_rf else 0
            xgb_output = 1 if y_pred_xgb[i] > threshold_xgb else 0

            y_test_pred.append(lstm_output)
            if lstm_output == 1 and y_test[i] == 1 and (rf_output == 0 or xgb_output == 0):
                num_decisions_correctly_taken_from_lstm_and_not_from_xgb_or_rf += 1
            if lstm_output == 1 and y_test[i]:
                num_decisions_correctly_taken_from_lstm += 1

        elif abs(y_pred_rf[i] - rf_mean) == max_value:
            y_test_pred.append(1 if y_pred_rf[i] > threshold_rf else 0)
            num_decisions_taken_by_rf += 1

        else:
            num_decisions_taken_by_xgb += 1
            y_test_pred.append(1 if y_pred_xgb[i] > threshold_xgb else 0)

    return y_test_pred, num_decisions_taken_by_lstm, num_decisions_taken_by_rf, num_decisions_taken_by_xgb, num_decisions_correctly_taken_from_lstm, num_decisions_correctly_taken_from_lstm_and_not_from_xgb_or_rf

# for each model, get threshold and get mean of the output on validation set .
# on test set, calculate the distance from the calculated mean.
# The model that has the max distance will be the one who decides only if his distance is greater than the sum of the other two distances.
# Otherwise, use the majority voting
def predict_test_based_on_more_confident_and_majority_voting(lstm, rf, xg_reg, x_val, x_val_supervised, y_val, x_test, x_test_supervised, y_test):
    y_val_pred_lstm, y_val_pred_rf, y_val_pred_xgb = get_predictions_for_each_model(lstm, rf, xg_reg, x_val, x_val_supervised)

    # get threshold for each model
    threshold_lstm = evaluation.find_best_threshold_fixed_fpr(y_val, y_val_pred_lstm)
    threshold_rf = evaluation.find_best_threshold_fixed_fpr(y_val, y_val_pred_rf)
    threshold_xgb = evaluation.find_best_threshold_fixed_fpr(y_val, y_val_pred_xgb)

    # get mean for each model
    lstm_mean = np.array(y_val_pred_lstm).mean()
    rf_mean = np.array(y_val_pred_rf).mean()
    xgb_mean = np.array(y_val_pred_xgb).mean()
    # get std for each model
    lstm_std = np.array(y_val_pred_lstm).std()
    rf_std = np.array(y_val_pred_rf).std()
    xgb_std = np.array(y_val_pred_xgb).std()

    # predicting test set
    y_pred_lstm, y_pred_rf, y_pred_xgb = get_predictions_for_each_model(lstm, rf, xg_reg, x_test, x_test_supervised)

    y_test_pred = []
    num_decisions_taken_by_lstm = 0
    num_decisions_correctly_taken_from_lstm = 0
    num_decisions_correctly_taken_from_lstm_and_not_from_xgb_or_rf = 0
    num_decisions_taken_by_rf = 0
    num_decisions_taken_by_xgb = 0
    for i in range(len(y_pred_lstm)):
        max_value = max(abs(y_pred_lstm[i] - lstm_mean)/lstm_std, abs(y_pred_rf[i] - rf_mean)/rf_std, abs(y_pred_xgb[i] - xgb_mean)/xgb_std)
        lstm_output = 1 if y_pred_lstm[i] > threshold_lstm else 0
        rf_output = 1 if y_pred_rf[i] > threshold_rf else 0
        xgb_output = 1 if y_pred_xgb[i] > threshold_xgb else 0
        if abs(y_pred_lstm[i] - lstm_mean)/lstm_std == max_value:
            if max_value > abs(y_pred_rf[i] - rf_mean)/rf_std + abs(y_pred_xgb[i] - xgb_mean)/xgb_std:
                num_decisions_taken_by_lstm += 1
                y_test_pred.append(lstm_output)
                if lstm_output == 1 and y_test[i] == 1 and (rf_output == 0 or xgb_output == 0):
                    num_decisions_correctly_taken_from_lstm_and_not_from_xgb_or_rf += 1
                if lstm_output == 1 and y_test[i]:
                    num_decisions_correctly_taken_from_lstm += 1
            else:
                if lstm_output + rf_output + xgb_output >= 2:
                    y_test_pred.append(1)
                else:
                    y_test_pred.append(0)

        elif abs(y_pred_rf[i] - rf_mean)/rf_std == max_value:
            if max_value > abs(y_pred_lstm[i] - lstm_mean)/lstm_std + abs(y_pred_xgb[i] - xgb_mean)/xgb_std:
                y_test_pred.append(1 if y_pred_rf[i] > threshold_rf else 0)
                num_decisions_taken_by_rf += 1
            else:
                if lstm_output + rf_output + xgb_output >= 2:
                    y_test_pred.append(1)
                else:
                    y_test_pred.append(0)
        else:
            if max_value > abs(y_pred_rf[i] - rf_mean)/rf_std + abs(y_pred_lstm[i] - lstm_mean)/lstm_std:
                num_decisions_taken_by_xgb += 1
                y_test_pred.append(1 if y_pred_xgb[i] > threshold_xgb else 0)
            else:
                if lstm_output + rf_output + xgb_output >= 2:
                    y_test_pred.append(1)
                else:
                    y_test_pred.append(0)
    return y_test_pred

# map each model output in a exp (the ouput shape of each model seems to follow this type of distribution) distribution
def predict_test_based_on_expon(lstm, rf, xg_reg, x_val, x_val_supervised, y_val, x_test, x_test_supervised, y_test):
    y_val_pred_lstm, y_val_pred_rf, y_val_pred_xgb = get_predictions_for_each_model(lstm, rf, xg_reg, x_val, x_val_supervised)

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

    # for each model, fit the exponential distribution
    loc_lstm, scale_lstm = stats.expon.fit(y_val_pred_lstm)
    loc_rf, scale_rf = stats.expon.fit(y_val_pred_rf)
    loc_xgb, scale_xgb = stats.expon.fit(y_val_pred_xgb)
    samples_lstm = stats.expon.cdf(y_val_pred_lstm, scale=scale_lstm, loc=loc_lstm)
    samples_rf = stats.expon.cdf(y_val_pred_rf, scale=scale_rf, loc=loc_rf)
    samples_xgb = stats.expon.cdf(y_val_pred_xgb, scale=scale_xgb, loc=loc_xgb)

    # get mean for each model
    lstm_mean = samples_lstm.mean()
    rf_mean = samples_rf.mean()
    xgb_mean = samples_xgb.mean()

    # get std for each model
    lstm_std = samples_lstm.std()
    rf_std = samples_rf.std()
    xgb_std = samples_xgb.std()

    threshold_lstm = evaluation.find_best_threshold_fixed_fpr(y_val, samples_lstm)
    threshold_rf = evaluation.find_best_threshold_fixed_fpr(y_val, samples_rf)
    threshold_xgb = evaluation.find_best_threshold_fixed_fpr(y_val, samples_xgb)

    y_pred_lstm, y_pred_rf, y_pred_xgb = get_predictions_for_each_model(lstm, rf, xg_reg, x_test, x_test_supervised)

    samples_lstm = stats.expon.cdf(y_pred_lstm, scale=scale_lstm, loc=loc_lstm)
    samples_rf = stats.expon.cdf(y_pred_rf, scale=scale_rf, loc=loc_rf)
    samples_xgb = stats.expon.cdf(y_pred_xgb, scale=scale_xgb, loc=loc_xgb)

    num_decisions_taken_by_lstm = 0
    num_decisions_correctly_taken_from_lstm = 0
    num_decisions_correctly_taken_from_lstm_and_not_from_xgb_or_rf = 0
    num_decisions_taken_by_rf = 0
    num_decisions_taken_by_xgb = 0
    y_test_pred = []
    for i in range(len(y_pred_lstm)):
        max_value = max(abs(samples_lstm[i] - lstm_mean) / lstm_std, abs(samples_rf[i] - rf_mean) / rf_std, abs(samples_xgb[i] - xgb_mean) / xgb_std)

        if abs(samples_lstm[i] - lstm_mean) / lstm_std == max_value:
            lstm_output = 1 if samples_lstm[i] > threshold_lstm else 0
            rf_output = 1 if samples_rf[i] > threshold_rf else 0
            xgb_output = 1 if samples_xgb[i] > threshold_xgb else 0

            if lstm_output == 1 and y_test[i] == 1 and (rf_output == 0 or xgb_output == 0) or lstm_output == 0 and y_test[i] == 0 and (rf_output == 1 or xgb_output == 1):
                num_decisions_correctly_taken_from_lstm_and_not_from_xgb_or_rf += 1

            if (lstm_output == 1 and y_test[i] == 1) or (lstm_output == 0 and y_test[i] == 0):
                num_decisions_correctly_taken_from_lstm += 1

            num_decisions_taken_by_lstm += 1
            y_test_pred.append(1 if samples_lstm[i] > threshold_lstm else 0)

        elif abs(samples_rf[i] - rf_mean) / rf_std == max_value:
            num_decisions_taken_by_rf += 1
            y_test_pred.append(1 if samples_rf[i] > threshold_rf else 0)

        elif abs(samples_xgb[i] - xgb_mean) / xgb_std == max_value:
            num_decisions_taken_by_xgb += 1
            y_test_pred.append(1 if samples_xgb[i] > threshold_xgb else 0)

        else:
            print("errore")

    '''
    loc_lstm, scale_lstm = stats.expon.fit(y_val_pred_lstm)
    loc_rf, scale_rf = stats.expon.fit(y_val_pred_rf)
    loc_xgb, scale_xgb = stats.expon.fit(y_val_pred_xgb)
    samples = stats.expon.cdf(y_val_pred_lstm, scale=scale_lstm, loc=loc_lstm) * np.array(len(y_val_pred_lstm) * [0.333])
    samples += stats.expon.cdf(y_val_pred_rf, scale=scale_rf, loc=loc_rf) * np.array(len(y_val_pred_lstm) * [0.333])
    samples += stats.expon.cdf(y_val_pred_xgb, scale=scale_xgb, loc=loc_xgb) * np.array(len(y_val_pred_lstm) * [0.333])
    threshold = evaluation.find_best_threshold_fixed_fpr(y_val, samples)

    y_pred_lstm, y_pred_rf, y_pred_xgb = get_predictions_for_each_model(lstm, rf, xg_reg, x_test, x_test_supervised)

    samples = stats.expon.cdf(y_pred_lstm, scale=scale_lstm, loc=loc_lstm) * np.array(len(y_pred_lstm) * [0.333])
    samples += stats.expon.cdf(y_pred_rf, scale=scale_rf, loc=loc_rf) * np.array(len(y_pred_lstm) * [0.333])
    samples += stats.expon.cdf(y_pred_xgb, scale=scale_xgb, loc=loc_xgb) * np.array(len(y_pred_lstm) * [0.333])
    y_test_pred = evaluation.adjusted_classes(samples, threshold)
    '''
    return y_test_pred, num_decisions_taken_by_lstm, num_decisions_taken_by_rf, num_decisions_taken_by_xgb, num_decisions_correctly_taken_from_lstm, num_decisions_correctly_taken_from_lstm_and_not_from_xgb_or_rf
# --------------------------- end of the types of ensembles ----------------------------

# if adversarial attack is set to true, craft adversarial frauds and get the performances
# NB: is_white_box_attack must be used to create an experiment that use as dataset: "REAL_DATASET"
# if is_white_box_attack == False, the dataset used will be "OLD_DATASET"
def repeat_experiment_n_times(lstm, rf, xg_reg, scenario, times_to_repeat=100, adversarial_attack=False, evasion_attack=False, is_white_box_attack=True, use_lstm_for_adversarial=False):
    tn_s = []
    tp_s = []
    fp_s = []
    fn_s = []
    f1_s = []
    balanced_accuracies = []
    precisions = []
    recalls = []
    aucpr_s = []
    roc_aucs = []

    num_decisions_taken_by_lstm,\
    num_decisions_taken_by_rf, \
    num_decisions_taken_by_xgb, \
    num_decisions_correctly_taken_from_lstm, \
    num_decisions_correctly_taken_from_lstm_and_not_from_xgb_or_rf = 0, 0, 0, 0, 0

    for i in range(times_to_repeat):
        print("Iteration", i)
        x_test_set, y_test_set = sequences_crafting_for_classification.get_test_set(scenario=scenario)

        x_val, y_val, x_test, y_test = evaluation.get_val_test_set(x_test_set, y_test_set, val_size=0.25)
        x_val_supervised = x_val[:, len(x_val[0]) - 1, :]
        x_test_supervised = x_test[:, len(x_val[0]) - 1, :]

        if adversarial_attack or evasion_attack:
            # getting train set for training
            if is_white_box_attack:
                print("Using as training set, the real one - whitebox attack")
                dataset_type = REAL_DATASET
            else:
                print("Using as training set, the old one - blackbox attack")
                dataset_type = OLD_DATASET

            x_train, y_train = sequences_crafting_for_classification.get_train_set(dataset_type=dataset_type)
            x_train_supervised = x_train[:, look_back, :]
            if adversarial_attack:
                print("Crafting an adversarial attack")
                if not use_lstm_for_adversarial:
                    print("The attacker will use a Multilayer perceptron")
                    # training multilayer perceptron
                    # todo: hyper param tuning multilayer perceptron
                    adversarial_model = MultiLayerPerceptron.create_fit_model(x_train_supervised, y_train)
                    # crafting adversarial samples
                    x_test_supervised = x_test[:, len(x_test[0]) - 1, :]
                    frauds = x_test_supervised[np.where(y_test == 1)]

                    adversarial_samples = fgsm.craft_sample(frauds, adversarial_model, epsilon=0.01)

                    x_test[np.where(y_test == 1), len(x_test[0]) - 1] = adversarial_samples
                    x_test_supervised = x_test[:, len(x_test[0]) - 1, :]
                else:
                    print("The attacker will use a LSTM network")
                    # train the network using the right params
                    if is_white_box_attack:
                        params = BEST_PARAMS_LSTM_REAL_DATASET
                    else:
                        params = BEST_PARAMS_LSTM_OLD_DATASET
                    adversarial_model = LSTM_classifier.create_fit_model(x_train, y_train, look_back, params=params)
                    frauds = x_test[np.where(y_test == 1)]
                    adversarial_samples = fgsm.craft_sample(frauds, adversarial_model, epsilon=0.01)
                    x_test[np.where(y_test == 1)] = adversarial_samples
                    x_test_supervised = x_test[:, len(x_test[0]) - 1, :]

            if evasion_attack:
                print("Crafting an evasion attack")
                # train the network using the right params
                if is_white_box_attack:
                    params = BEST_PARAMS_RF
                else:
                    params = BEST_PARAMS_RF_OLD_DATASET
                # training the oracle
                oracle = RF.create_model(x_train_supervised, y_train, params=params)

                # get the oracle threshold
                y_val_pred_oracle = oracle.predict_proba(x_val_supervised)
                oracle_threshold = evaluation.find_best_threshold_fixed_fpr(y_val, y_val_pred_oracle[:, 1])

                # if the oracle predicts the fraud as fraud -> discard it, otherwise inject in real bank system
                y_pred_oracle = rf.predict_proba(x_test_supervised)
                y_pred_oracle = y_pred_oracle[:, 1].ravel()
                y_pred_oracle = np.array(evaluation.adjusted_classes(y_pred_oracle, oracle_threshold))

                x_test = x_test[(np.where(((y_test == 1) & (y_pred_oracle == 0)) | (y_test == 0)))]
                y_test = y_test[(np.where(((y_test == 1) & (y_pred_oracle == 0)) | (y_test == 0)))]
                x_test_supervised = x_test[:, len(x_test[0]) - 1, :]
        try:
            # a, b, c, d, e = 0, 0, 0, 0, 0
            # y_test_pred, not_by_xgb, not_by_rf, not_found_by_others = predict_test_based_on_voting(lstm, rf, xg_reg, x_val, x_val_supervised, y_val, x_test, x_test_supervised, y_test)
            # y_test_pred, a, b, c, d, e = predict_test_based_on_more_confident(lstm, rf, xg_reg, x_val, x_val_supervised, y_val, x_test, x_test_supervised, y_test)
            y_test_pred, a, b, c, d, e = predict_test_based_on_expon(lstm, rf, xg_reg, x_val, x_val_supervised, y_val, x_test, x_test_supervised, y_test)
            # y_test_pred = predict_test_based_on_sum(lstm, rf, xg_reg, x_val, x_val_supervised, y_val, x_test, x_test_supervised)
            # y_test_pred = predict_test_based_on_more_confident_and_majority_voting(lstm, rf, xg_reg, x_val, x_val_supervised, y_val, x_test, x_test_supervised, y_test)
            # not_found_by_xgboost += not_by_xgb
            # not_by_rf += not_by_rf
            # not_found_by_others += not_by_others

            y_test_pred = np.array(y_test_pred)
            confusion, f1, balanced_accuracy, precision, recall, aucpr, roc_auc = evaluation.get_performance(y_test, y_test_pred, threshold=True)
            tn = confusion[0, 0]
            tp = confusion[1, 1]
            fp = confusion[0, 1]
            fn = confusion[1, 0]

            tn_s.append(tn)
            tp_s.append(tp)
            fp_s.append(fp)
            fn_s.append(fn)
            f1_s.append(f1)

            num_decisions_taken_by_lstm += a
            num_decisions_taken_by_rf += b
            num_decisions_taken_by_xgb += c
            num_decisions_correctly_taken_from_lstm += d
            num_decisions_correctly_taken_from_lstm_and_not_from_xgb_or_rf += e

            balanced_accuracies.append(balanced_accuracy)
            precisions.append(precision)
            recalls.append(recall)
            aucpr_s.append(aucpr)
            roc_aucs.append(roc_auc)
        except RuntimeError:
            i -= 1

    print("Num decisions taken from lstm: ", num_decisions_taken_by_lstm / times_to_repeat)
    print("Num decisions taken by rf: ", num_decisions_taken_by_rf / times_to_repeat)
    print("Num decisions taken by xgb: ", num_decisions_taken_by_xgb / times_to_repeat)
    print("Num decisions taken by lstm correctly taken: ", num_decisions_correctly_taken_from_lstm / times_to_repeat)
    print("Num decisions taken by lstm correctly taken and not by others: ", num_decisions_correctly_taken_from_lstm_and_not_from_xgb_or_rf / times_to_repeat)
    evaluation.print_results(np.array(tn_s).mean(), np.array(fp_s).mean(), np.array(fn_s).mean(), np.array(tp_s).mean(), np.array(f1_s).mean(), np.array(balanced_accuracies).mean(), np.array(precisions).mean(), np.array(recalls).mean(), np.array(aucpr_s).mean(), np.array(roc_aucs).mean())


look_back = LOOK_BACK
print("Lookback using: ", look_back)
x_train, y_train = sequences_crafting_for_classification.get_train_set()
# train model for supervised models (xgboost/rf)
x_train_supervised = x_train[:, look_back, :]
y_train_supervised = y_train

print("Training models...")
lstm = LSTM_classifier.create_fit_model(x_train, y_train, look_back, params={'layers': {'input': 64, 'hidden1': 64, 'output': 1}, 'epochs': 10, 'dropout_rate': 0.3, 'batch_size': 32})
rf = RF.create_model(x_train_supervised, y_train_supervised)
xg_reg = xgboost_classifier.create_model(x_train_supervised, y_train_supervised)

if DATASET_TYPE == INJECTED_DATASET or DATASET_TYPE == OLD_DATASET:
    scenarios = [FIRST_SCENARIO, SECOND_SCENARIO, THIRD_SCENARIO, FOURTH_SCENARIO, FIFTH_SCENARIO, SIXTH_SCENARIO, SEVENTH_SCENARIO, EIGHTH_SCENARIO, NINTH_SCENARIO, ALL_SCENARIOS]
    scenarios = [ALL_SCENARIOS]
    for scenario in scenarios:
        print("-------------------", scenario, "scenario --------------------------")
        repeat_experiment_n_times(lstm, rf, xg_reg, scenario, times_to_repeat=10, adversarial_attack=False, evasion_attack=False, is_white_box_attack=False, use_lstm_for_adversarial=True)

if DATASET_TYPE == REAL_DATASET:
    repeat_experiment_n_times(lstm, rf, xg_reg, False, times_to_repeat=10, adversarial_attack=False, evasion_attack=True, is_white_box_attack=True, use_lstm_for_adversarial=True)