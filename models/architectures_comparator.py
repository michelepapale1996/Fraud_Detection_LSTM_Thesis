import numpy as np
from sklearn.ensemble import RandomForestClassifier
from dataset_creation import sequences_crafting_for_classification
from dataset_creation.constants import *
from models import MultiLayerPerceptron, LSTM_classifier, RF, xgboost_classifier, evaluation
from adversarial_attacks import fgsm

# adversarial attack creates frauds starting from the frauds and using FGSM
# evasion attack uses as frauds only the one non detected by an oracle
# is_white_box_attack decides which dataset will be used (if 2013/2014 or 2014/2015) for training the oracle for evasion/adversarial attack
def repeat_experiment_n_times(lstm, rf, xg_reg, scenario, times_to_repeat=100, adversarial_attack=False, evasion_attack=False, is_white_box_attack=True, use_lstm_for_adversarial=False):
    x_test_real, y_test_real = sequences_crafting_for_classification.get_test_set(scenario=scenario)

    if adversarial_attack or evasion_attack:
        lstm_tn_s = []
        lstm_tp_s = []
        lstm_fp_s = []
        lstm_fn_s = []
        lstm_precisions = []
        lstm_recalls = []

        rf_tn_s = []
        rf_tp_s = []
        rf_fp_s = []
        rf_fn_s = []
        rf_precisions = []
        rf_recalls = []

        xgb_tn_s = []
        xgb_tp_s = []
        xgb_fp_s = []
        xgb_fn_s = []
        xgb_precisions = []
        xgb_recalls = []

        # getting train set for training
        if is_white_box_attack:
            print("Using as traing set, the real one - whitebox attack")
            dataset_type = REAL_DATASET
        else:
            print("Using as traing set, the old one - blackbox attack")
            dataset_type = OLD_DATASET

        x_train, y_train = sequences_crafting_for_classification.get_train_set(dataset_type=dataset_type)
        x_train_supervised = x_train[:, look_back, :]

        for i in range(times_to_repeat):
            x_val, y_val, x_test, y_test = evaluation.get_val_test_set(x_test_real, y_test_real, val_size=0.25)
            x_val_supervised = x_val[:, len(x_val[0]) - 1, :]

            y_val_pred_lstm = lstm.predict(x_val)
            y_val_pred_rf = rf.predict_proba(x_val_supervised)
            y_val_pred_xgb = xg_reg.predict_proba(x_val_supervised)

            # get threshold for each model
            threshold_lstm = evaluation.find_best_threshold_fixed_fpr(y_val, y_val_pred_lstm.ravel())
            threshold_rf = evaluation.find_best_threshold_fixed_fpr(y_val, y_val_pred_rf[:, 1])
            threshold_xgb = evaluation.find_best_threshold_fixed_fpr(y_val, y_val_pred_xgb[:, 1])

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
                        params = BEST_PARAMS_LSTM_REAL_DATASET
                    else:
                        params = BEST_PARAMS_LSTM_OLD_DATASET
                    frauds = x_test[np.where(y_test == 1)]
                    adversarial_model = LSTM_classifier.create_fit_model(x_train, y_train, look_back, params=params)
                    adversarial_samples = fgsm.craft_sample(frauds, adversarial_model, epsilon=0.01)
                    # in lstm samples, must be changed the last transaction of the sequence
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

            confusion, f1, balanced_accuracy, precision, recall, aucpr, roc_auc = evaluation.get_performance(y_test, y_pred_lstm, threshold=threshold_lstm)
            lstm_tn_s.append(confusion[0, 0])
            lstm_tp_s.append(confusion[1, 1])
            lstm_fp_s.append(confusion[0, 1])
            lstm_fn_s.append(confusion[1, 0])
            lstm_precisions.append(precision)
            lstm_recalls.append(recall)

            confusion, f1, balanced_accuracy, precision, recall, aucpr, roc_auc = evaluation.get_performance(y_test, y_pred_rf, threshold=threshold_rf)
            rf_tn_s.append(confusion[0, 0])
            rf_tp_s.append(confusion[1, 1])
            rf_fp_s.append(confusion[0, 1])
            rf_fn_s.append(confusion[1, 0])
            rf_precisions.append(precision)
            rf_recalls.append(recall)

            confusion, f1, balanced_accuracy, precision, recall, aucpr, roc_auc = evaluation.get_performance(y_test, y_pred_xgb, threshold=threshold_xgb)
            xgb_tn_s.append(confusion[0, 0])
            xgb_tp_s.append(confusion[1, 1])
            xgb_fp_s.append(confusion[0, 1])
            xgb_fn_s.append(confusion[1, 0])
            xgb_precisions.append(precision)
            xgb_recalls.append(recall)

        print("LSTM")
        evaluation.print_results(np.array(lstm_tn_s).mean(), np.array(lstm_fp_s).mean(), np.array(lstm_fn_s).mean(), np.array(lstm_tp_s).mean(), 0, 0, np.array(lstm_precisions).mean(), np.array(lstm_recalls).mean(), 0, 0)
        print("RF")
        evaluation.print_results(np.array(rf_tn_s).mean(), np.array(rf_fp_s).mean(), np.array(rf_fn_s).mean(),
                                 np.array(rf_tp_s).mean(), 0, 0, np.array(rf_precisions).mean(),
                                 np.array(rf_recalls).mean(), 0, 0)
        print("XGB")
        evaluation.print_results(np.array(xgb_tn_s).mean(), np.array(xgb_fp_s).mean(), np.array(xgb_fn_s).mean(), np.array(xgb_tp_s).mean(), 0, 0, np.array(xgb_precisions).mean(), np.array(xgb_recalls).mean(), 0, 0)

    if not adversarial_attack and not evasion_attack:
        x_test_supervised = x_test_real[:, len(x_test_real[0]) - 1, :]

        print("LSTM")
        y_pred_lstm = lstm.predict(x_test_real)
        evaluation.evaluate_n_times(y_test_real, y_pred_lstm, times_to_repeat=times_to_repeat)

        print("RF")
        y_pred_rf = rf.predict_proba(x_test_supervised)
        evaluation.evaluate_n_times(y_test_real, y_pred_rf[:, 1], times_to_repeat=times_to_repeat)

        print("Xgboost")
        y_pred_xgb = xg_reg.predict_proba(x_test_supervised)
        evaluation.evaluate_n_times(y_test_real, y_pred_xgb[:, 1], times_to_repeat=times_to_repeat)

        evaluation.print_jaccard_index(lstm, rf, xg_reg, x_test_real, y_test_real, look_back)


look_back = LOOK_BACK
print("Using as lookback:", look_back)
x_train, y_train = sequences_crafting_for_classification.get_train_set()
# train model for supervised models (xgboost/rf)
x_train_supervised = x_train[:, look_back, :]
y_train_supervised = y_train

print("Training models...")
lstm = LSTM_classifier.create_fit_model(x_train, y_train, look_back, params={'layers': {'input': 64, 'hidden1': 64, 'output': 1}, 'epochs': 10, 'dropout_rate': 0.3, 'batch_size': 32})
xg_reg = xgboost_classifier.create_model(x_train_supervised, y_train_supervised)
rf = RF.create_model(x_train_supervised, y_train_supervised)

if DATASET_TYPE == INJECTED_DATASET or DATASET_TYPE == OLD_DATASET:
    scenarios = [FIRST_SCENARIO, SECOND_SCENARIO, THIRD_SCENARIO, FOURTH_SCENARIO, FIFTH_SCENARIO, SIXTH_SCENARIO, SEVENTH_SCENARIO, EIGHTH_SCENARIO, NINTH_SCENARIO, ALL_SCENARIOS]
    scenarios = [ALL_SCENARIOS]
    for scenario in scenarios:
        print("-------------------", scenario, "scenario --------------------------")
        repeat_experiment_n_times(lstm, rf, xg_reg, scenario=scenario, times_to_repeat=10, adversarial_attack=False, evasion_attack=False, is_white_box_attack=False, use_lstm_for_adversarial=True)

if DATASET_TYPE == REAL_DATASET:
    repeat_experiment_n_times(lstm, rf, xg_reg, scenario=False, times_to_repeat=10, adversarial_attack=False, evasion_attack=True, is_white_box_attack=True, use_lstm_for_adversarial=True)