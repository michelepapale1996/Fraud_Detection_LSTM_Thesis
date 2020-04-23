import numpy as np
from sklearn.ensemble import RandomForestClassifier
from dataset_creation import sequences_crafting_for_classification
from dataset_creation.constants import *
from models import MultiLayerPerceptron, LSTM_classifier, RF, xgboost_classifier, evaluation, explainability, resampling_dataset
from adversarial_attacks import fgsm

# adversarial attack creates frauds starting from the frauds and using FGSM
# evasion attack uses as frauds only the one non detected by an oracle
# is_white_box_attack decides which dataset will be used (if 2013/2014 or 2014/2015) for training the oracle for evasion/adversarial attack
def experiment(lstm, threshold_lstm, xg_reg, threshold_xgb, rf, threshold_rf, scenario, adversarial_attack=False, evasion_attack=False, is_white_box_attack=True, use_lstm_for_adversarial=False):
    x_test, y_test = sequences_crafting_for_classification.get_test_set(scenario=scenario)

    if adversarial_attack or evasion_attack:
        # getting train set for training
        if is_white_box_attack:
            print("Using as traing set, the real one - whitebox attack")
            dataset_type = INJECTED_DATASET
        else:
            print("Using as traing set, the old one - blackbox attack")
            dataset_type = OLD_DATASET

        x_train, y_train = sequences_crafting_for_classification.get_train_set(dataset_type=dataset_type)
        x_train_supervised = x_train[:, look_back, :]

        x_test_supervised = x_test[:, len(x_test[0]) - 1, :]
        if adversarial_attack:
            print("Crafting an adversarial attack")
            if not use_lstm_for_adversarial:
                print("The attacker will use a Multilayer perceptron")
                adversarial_model = MultiLayerPerceptron.create_fit_model(x_train_supervised, y_train)
                frauds = x_test_supervised[np.where(y_test == 1)]
                adversarial_samples = fgsm.craft_sample(frauds, adversarial_model, epsilon=0.01)
                # in lstm samples, must be changed the last transaction of the sequence
                x_test[np.where(y_test == 1), len(x_test[0]) - 1] = adversarial_samples
                x_test_supervised = x_test[:, len(x_test[0]) - 1, :]
            else:
                print("The attacker will use a LSTM network")
                # train the network using the right params
                if is_white_box_attack:
                    if USING_AGGREGATED_FEATURES:
                        params = BEST_PARAMS_LSTM_REAL_DATASET_AGGREGATED
                    else:
                        params = BEST_PARAMS_LSTM_REAL_DATASET_NO_AGGREGATED
                else:
                    if USING_AGGREGATED_FEATURES:
                        params = BEST_PARAMS_LSTM_OLD_DATASET_AGGREGATED
                    else:
                        params = BEST_PARAMS_LSTM_OLD_DATASET_NO_AGGREGATED
                frauds = x_test[np.where(y_test == 1)]
                adversarial_model, _ = LSTM_classifier.create_fit_model(x_train, y_train, look_back, params=params)
                adversarial_samples = fgsm.craft_sample(frauds, adversarial_model, epsilon=0.1)
                # in lstm samples, must be changed the last transaction of the sequence
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
            y_pred_oracle = rf.predict_proba(x_test_supervised)
            y_pred_oracle = y_pred_oracle[:, 1].ravel()
            y_pred_oracle = np.array(evaluation.adjusted_classes(y_pred_oracle, oracle_threshold))

            x_test = x_test[(np.where((y_test == 1) & (y_pred_oracle == 0) | (y_test == 0)))]
            y_test = y_test[(np.where((y_test == 1) & (y_pred_oracle == 0) | (y_test == 0)))]
            x_test_supervised = x_test[:, len(x_test[0]) - 1, :]

        # predicting test set
        y_pred_lstm = lstm.predict(x_test)
        y_pred_rf = rf.predict_proba(x_test_supervised)
        y_pred_xgb = xg_reg.predict_proba(x_test_supervised)
        y_pred_lstm = y_pred_lstm.ravel()
        y_pred_rf = y_pred_rf[:, 1].ravel()
        y_pred_xgb = y_pred_xgb[:, 1].ravel()

        print("LSTM")
        evaluation.evaluate(y_test, y_pred_lstm, threshold_lstm)

        print("RF")
        evaluation.evaluate(y_test, y_pred_rf, threshold_rf)

        print("Xgboost")
        evaluation.evaluate(y_test, y_pred_xgb, threshold_xgb)

    if not adversarial_attack and not evasion_attack:
        x_test_supervised = x_test[:, len(x_test[0]) - 1, :]

        x_train, y_train = sequences_crafting_for_classification.get_train_set()
        x_train_supervised = x_train[:, look_back, :]

        print("LSTM")
        y_pred_lstm = lstm.predict(x_test)
        evaluation.evaluate(y_test, y_pred_lstm, threshold_lstm)
        explainability.explain_dataset(lstm, x_train, x_test, threshold_lstm, y_test)

        print("RF")
        y_pred_rf = rf.predict_proba(x_test_supervised)[:, 1]
        evaluation.evaluate(y_test, y_pred_rf, threshold_rf)
        explainability.explain_dataset(rf, x_train_supervised, x_test_supervised, threshold_rf, y_test)

        print("Xgboost")
        y_pred_xgb = xg_reg.predict_proba(x_test_supervised)[:, 1]
        evaluation.evaluate(y_test, y_pred_xgb, threshold_xgb)
        explainability.explain_dataset(xg_reg, x_train_supervised, x_test_supervised, threshold_xgb, y_test)

    y_pred_lstm = evaluation.adjusted_classes(y_pred_lstm, threshold_lstm)
    y_pred_rf = evaluation.adjusted_classes(y_pred_rf, threshold_rf)
    y_pred_xgb = evaluation.adjusted_classes(y_pred_xgb, threshold_xgb)

    lstm_fraud_indices = evaluation.get_fraud_indices(y_test, y_pred_lstm)
    rf_fraud_indices = evaluation.get_fraud_indices(y_test, y_pred_rf)
    xgboost_fraud_indices = evaluation.get_fraud_indices(y_test, y_pred_xgb)
    evaluation.print_frauds_stats(lstm_fraud_indices, rf_fraud_indices, xgboost_fraud_indices)

    lstm_genuine_indices = evaluation.get_genuine_indices(y_test, y_pred_lstm)
    rf_genuine_indices = evaluation.get_genuine_indices(y_test, y_pred_rf)
    xgboost_genuine_indices = evaluation.get_genuine_indices(y_test, y_pred_xgb)
    evaluation.print_genuine_stats(lstm_genuine_indices, rf_genuine_indices, xgboost_genuine_indices)


look_back = LOOK_BACK
print("Using as lookback:", look_back)

x_train, y_train = sequences_crafting_for_classification.get_train_set()
# if the dataset is the real one -> contrast imbalanced dataset problem
if DATASET_TYPE == REAL_DATASET:
    x_train, y_train = resampling_dataset.oversample_set(x_train, y_train)

# train model for supervised models (xgboost/rf)
x_train_supervised = x_train[:, look_back, :]
y_train_supervised = y_train

print("Training models...")
times_to_repeat = 10
lstm, threshold_lstm = LSTM_classifier.create_fit_model(x_train, y_train, look_back, times_to_repeat=times_to_repeat)
xg_reg, threshold_xgb = xgboost_classifier.create_model(x_train_supervised, y_train_supervised, times_to_repeat=times_to_repeat)
rf, threshold_rf = RF.create_model(x_train_supervised, y_train_supervised, times_to_repeat=times_to_repeat)

if DATASET_TYPE == INJECTED_DATASET or DATASET_TYPE == OLD_DATASET:
    scenarios = [FIRST_SCENARIO, SECOND_SCENARIO, THIRD_SCENARIO, FOURTH_SCENARIO, FIFTH_SCENARIO, SIXTH_SCENARIO, SEVENTH_SCENARIO, EIGHTH_SCENARIO, NINTH_SCENARIO, ALL_SCENARIOS]
    # scenarios = [ALL_SCENARIOS]
    for scenario in scenarios:
        print("-------------------", scenario, "scenario --------------------------")
        experiment(lstm, threshold_lstm, xg_reg, threshold_xgb, rf, threshold_rf, scenario=scenario, adversarial_attack=False, evasion_attack=False, is_white_box_attack=True, use_lstm_for_adversarial=True)

if DATASET_TYPE == REAL_DATASET:
    experiment(lstm, threshold_lstm, xg_reg, threshold_xgb, rf, threshold_rf, scenario=False, adversarial_attack=False, evasion_attack=False, is_white_box_attack=True, use_lstm_for_adversarial=True)