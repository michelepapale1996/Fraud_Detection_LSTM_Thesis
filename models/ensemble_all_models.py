from models import LSTM_classifier, evaluation
import numpy as np
from dataset_creation import sequences_crafting_for_classification
from dataset_creation.constants import *
from adversarial_attacks import fgsm
from models import MultiLayerPerceptron, xgboost_classifier, RF, resampling_dataset
from scipy import stats

def get_predictions_for_each_model(lstm, rf, xg_reg, x, x_supervised):
    y_lstm = lstm.predict(x)
    y_rf = rf.predict_proba(x_supervised)
    y_xgb = xg_reg.predict_proba(x_supervised)
    return y_lstm.ravel(), y_rf[:, 1], y_xgb[:,1]

# map each model output in a exp (the ouput shape of each model seems to follow this type of distribution) distribution
def predict_test_based_on_expon(lstm, scale_lstm, loc_lstm, lstm_mean, lstm_std, threshold_lstm, rf, scale_rf, loc_rf, rf_mean, rf_std, threshold_rf, xg_reg, scale_xgb, loc_xgb, xgb_mean, xgb_std, threshold_xgb, x_test, x_test_supervised, y_test):
    y_pred_lstm, y_pred_rf, y_pred_xgb = get_predictions_for_each_model(lstm, rf, xg_reg, x_test, x_test_supervised)

    samples_lstm = stats.expon.cdf(y_pred_lstm, scale=scale_lstm, loc=loc_lstm)
    samples_rf = stats.expon.cdf(y_pred_rf, scale=scale_rf, loc=loc_rf)
    samples_xgb = stats.expon.cdf(y_pred_xgb, scale=scale_xgb, loc=loc_xgb)

    num_decisions_taken_by_lstm = 0
    num_decisions_taken_by_rf = 0
    num_decisions_taken_by_xgb = 0
    y_test_pred = []
    thresholds = []
    for i in range(len(y_pred_lstm)):
        max_value = max(abs(samples_lstm[i] - lstm_mean) / lstm_std, abs(samples_rf[i] - rf_mean) / rf_std, abs(samples_xgb[i] - xgb_mean) / xgb_std)

        if abs(samples_lstm[i] - lstm_mean) / lstm_std == max_value:
            num_decisions_taken_by_lstm += 1
            y_test_pred.append(samples_lstm[i])
            thresholds.append(threshold_lstm)

        elif abs(samples_rf[i] - rf_mean) / rf_std == max_value:
            num_decisions_taken_by_rf += 1
            y_test_pred.append(samples_rf[i])
            thresholds.append(threshold_rf)

        elif abs(samples_xgb[i] - xgb_mean) / xgb_std == max_value:
            num_decisions_taken_by_xgb += 1
            y_test_pred.append(samples_xgb[i])
            thresholds.append(threshold_xgb)

        else:
            print("errore")

    return y_test_pred, thresholds, num_decisions_taken_by_lstm, num_decisions_taken_by_rf, num_decisions_taken_by_xgb

# --------------------------- end of the types of ensembles ----------------------------
def experiment_with_cdf(lstm, scale_lstm, loc_lstm, mean_lstm, std_lstm, threshold_lstm, rf, scale_rf, loc_rf, mean_rf, std_rf, threshold_rf, xg_reg, scale_xgb, loc_xgb, mean_xgb, std_xgb, threshold_xgb, scenario, adversarial_attack=False, evasion_attack=False, is_white_box_attack=True, use_lstm_for_adversarial=False):
    x_test, y_test = sequences_crafting_for_classification.get_test_set(scenario=scenario)
    x_test_supervised = x_test[:, len(x_test[0]) - 1, :]

    if adversarial_attack or evasion_attack:
        # getting train set for training
        if is_white_box_attack:
            print("whitebox attack")
            dataset_type = INJECTED_DATASET
        else:
            print("blackbox attack")
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
                    if USING_AGGREGATED_FEATURES:
                        params = BEST_PARAMS_LSTM_AGGREGATED
                    else:
                        params = BEST_PARAMS_LSTM_NO_AGGREGATED
                else:
                    if USING_AGGREGATED_FEATURES:
                        params = BEST_PARAMS_LSTM_OLD_DATASET_AGGREGATED
                    else:
                        params = BEST_PARAMS_LSTM_OLD_DATASET_NO_AGGREGATED
                adversarial_model, _ = LSTM_classifier.create_fit_model(x_train, y_train, look_back, params=params)
                frauds = x_test[np.where(y_test == 1)]
                adversarial_samples = fgsm.craft_sample(frauds, adversarial_model, epsilon=0.1)
                x_test[np.where(y_test == 1)] = adversarial_samples
                x_test_supervised = x_test[:, len(x_test[0]) - 1, :]

        if evasion_attack:
            print("Crafting an evasion attack")
            # train the network using the right params
            if is_white_box_attack:
                if USING_AGGREGATED_FEATURES:
                    params = BEST_PARAMS_RF_AGGREGATED
                else:
                    params = BEST_PARAMS_RF_NO_AGGREGATED
            else:
                if USING_AGGREGATED_FEATURES:
                    params = BEST_PARAMS_RF_OLD_DATASET_AGGREGATED
                else:
                    params = BEST_PARAMS_RF_OLD_DATASET_NO_AGGREGATED
            # training the oracle
            oracle, oracle_threshold = RF.create_model(x_train_supervised, y_train, params=params)

            # if the oracle predicts the fraud as fraud -> discard it, otherwise inject in real bank system
            y_pred_oracle = oracle.predict_proba(x_test_supervised)
            y_pred_oracle = y_pred_oracle[:, 1].ravel()
            y_pred_oracle = np.array(evaluation.adjusted_classes(y_pred_oracle, oracle_threshold))

            x_test = x_test[(np.where(((y_test == 1) & (y_pred_oracle == 0)) | (y_test == 0)))]
            y_test = y_test[(np.where(((y_test == 1) & (y_pred_oracle == 0)) | (y_test == 0)))]
            x_test_supervised = x_test[:, len(x_test[0]) - 1, :]

    y_test_pred, thresholds, num_decisions_taken_by_lstm, num_decisions_taken_by_rf, num_decisions_taken_by_xgb = predict_test_based_on_expon(lstm, scale_lstm, loc_lstm, mean_lstm, std_lstm, threshold_lstm, rf, scale_rf, loc_rf, mean_rf, std_rf, threshold_rf, xg_reg, scale_xgb, loc_xgb, mean_xgb, std_xgb, threshold_xgb, x_test, x_test_supervised, y_test)
    y_test_pred = np.array(y_test_pred)
    confusion, f1, balanced_accuracy, precision, recall, aucpr, roc_auc, fpr_values, tpr_values, accuracy, matthews_coeff = evaluation.get_performance(y_test, y_test_pred, thresholds)
    tn = confusion[0, 0]
    tp = confusion[1, 1]
    fp = confusion[0, 1]
    fn = confusion[1, 0]

    print("Num decisions taken from lstm: ", num_decisions_taken_by_lstm)
    print("Num decisions taken by rf: ", num_decisions_taken_by_rf)
    print("Num decisions taken by xgb: ", num_decisions_taken_by_xgb)
    evaluation.print_results(tn, fp, fn, tp, f1, balanced_accuracy, precision, recall, aucpr, roc_auc, fpr_values, tpr_values, accuracy, matthews_coeff)

look_back = LOOK_BACK
print("Lookback using: ", look_back)
print("MAX_FPR_RATE: ", MAX_FPR_RATE)
x_train, y_train = sequences_crafting_for_classification.get_train_set()

# if the dataset is the real one -> contrast imbalanced dataset problem
if DATASET_TYPE == REAL_DATASET:
    x_train, y_train = resampling_dataset.oversample_set(x_train, y_train)

# train model for supervised models (xgboost/rf)
x_train_supervised = x_train[:, look_back, :]
y_train_supervised = y_train

times_to_repeat = 10
print("Training models...")

lstm, scale_lstm, loc_lstm, mean_lstm, std_lstm, threshold_lstm = LSTM_classifier.create_fit_model_for_ensemble_based_on_cdf(x_train, y_train, look_back, times_to_repeat=times_to_repeat)
rf, scale_rf, loc_rf, mean_rf, std_rf, threshold_rf = RF.create_fit_model_for_ensemble_based_on_cdf(x_train_supervised, y_train_supervised, times_to_repeat=times_to_repeat)
xg_reg, scale_xgb, loc_xgb, mean_xgb, std_xgb, threshold_xgb = xgboost_classifier.create_fit_model_for_ensemble_based_on_cdf(x_train_supervised, y_train_supervised, times_to_repeat=times_to_repeat)

if DATASET_TYPE == INJECTED_DATASET or DATASET_TYPE == OLD_DATASET:
    scenarios = [FIRST_SCENARIO, SECOND_SCENARIO, THIRD_SCENARIO, FOURTH_SCENARIO, FIFTH_SCENARIO, SIXTH_SCENARIO, SEVENTH_SCENARIO, EIGHTH_SCENARIO, NINTH_SCENARIO, ALL_SCENARIOS]
    # scenarios = [ALL_SCENARIOS]
    for scenario in scenarios:
        print("-------------------", scenario, "scenario --------------------------")
        experiment_with_cdf(lstm, scale_lstm, loc_lstm, mean_lstm, std_lstm, threshold_lstm, rf, scale_rf, loc_rf, mean_rf, std_rf, threshold_rf, xg_reg, scale_xgb, loc_xgb, mean_xgb, std_xgb, threshold_xgb, scenario, adversarial_attack=False, evasion_attack=False, is_white_box_attack=True, use_lstm_for_adversarial=False)

if DATASET_TYPE == REAL_DATASET:
    experiment_with_cdf(lstm, scale_lstm, loc_lstm, mean_lstm, std_lstm, threshold_lstm, rf, scale_rf, loc_rf, mean_rf, std_rf, threshold_rf, xg_reg, scale_xgb, loc_xgb, mean_xgb, std_xgb, threshold_xgb, False, adversarial_attack=False, evasion_attack=False, is_white_box_attack=True, use_lstm_for_adversarial=False)
