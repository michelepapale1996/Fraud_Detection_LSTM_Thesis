from sklearn.linear_model import Ridge
import numpy as np
from models import evaluation
from keras.models import Sequential
import lime
import lime.lime_tabular
from dataset_creation import constants

def get_features():
    if constants.USING_AGGREGATED_FEATURES:
        feature_names = ['Importo', 'MsgErrore', 'NumConfermaSMS', 'isItalianSender',
                         'isItalianReceiver', 'daytime_x', 'daytime_y', 'during_hours_of_light',
                         'during_dark_hours', 'is_bonifico', 'time_delta',
                         'count_different_iban_ccs_1_window', 'count_different_asn_ccs_1_window',
                         'mean_amount_1_window', 'stdev_amount_1_window',
                         'count_trx_iban_1_window', 'is_new_asn_cc_1_window',
                         'is_new_iban_cc_1_window', 'count_different_iban_ccs_7_window',
                         'count_different_asn_ccs_7_window', 'mean_amount_7_window',
                         'stdev_amount_7_window', 'count_trx_iban_7_window',
                         'is_new_asn_cc_7_window', 'is_new_iban_cc_7_window']
    else:
        feature_names = ['Importo', 'MsgErrore', 'NumConfermaSMS', 'isItalianSender', 'isItalianReceiver', 'time_delta']
    return feature_names

def check_faithfulness(model, x, frauds_indices_found_by_model):
    if type(model) != Sequential:
        shape = len(x[0])
    else:
        shape = (x.shape[1], x.shape[2])

    feature_names = get_features()
    to_show = dict(zip(feature_names, [0] * len(feature_names)))
    print("There are", len(frauds_indices_found_by_model[0]), "frauds")
    for ith_fraud in range(100):  # len(frauds[0])):
        print(ith_fraud)
        fraud_index = frauds_indices_found_by_model[0][ith_fraud]
        x_to_study = x[fraud_index]

        x_sampled = []
        y_sampled = []
        for i in range(1000):

            noise = np.random.normal(0, 0.001, shape)
            perturbation = np.array(x_to_study - noise)
            if type(model) != Sequential:
                x_sampled.append(perturbation)
                perturbation = np.reshape(perturbation, (1, perturbation.shape[0]))
                y_ = model.predict_proba(perturbation)[0, 1]
            else:
                x_sampled.append(perturbation[:, len(perturbation) - 1])
                perturbation = np.reshape(perturbation, (1, perturbation.shape[0], -1))
                y_ = model.predict(perturbation)[0][0]

            y_sampled.append(y_)

        clf = Ridge(alpha=1.0)
        clf.fit(x_sampled, y_sampled)
        dictionary = dict(zip(feature_names, abs(clf.coef_)))
        top_5 = sorted(dictionary.items(), key=lambda x_: x_[1], reverse=True)[0:5]
        for elem in range(len(top_5)):
            to_show[top_5[elem][0]] += 1

    print(sorted(to_show.items(), key=lambda x_: x_[1], reverse=True))


def explain_frauds(model, x_train, x_test, frauds_indices_found_by_model):
    feature_names = get_features()
    to_show = dict(zip(feature_names, [0] * len(feature_names)))
    print("There are", len(frauds_indices_found_by_model), "frauds found by the model")
    if type(model) != Sequential:
        explainer = lime.lime_tabular.LimeTabularExplainer(x_train, feature_names=feature_names, class_names=["Genuine", "Fraud"], discretize_continuous=True)
        for ith_fraud in range(10): #range(len(frauds_indices_found_by_model)):
            print(ith_fraud)
            exp = explainer.explain_instance(x_test[frauds_indices_found_by_model[ith_fraud]], model.predict_proba, num_features=10, top_labels=2)
            for elem in exp.local_exp[0]:
                to_show[feature_names[elem[0]]] += 1
    else:
        explainer = lime.lime_tabular.RecurrentTabularExplainer(x_train, feature_names=feature_names, discretize_continuous=True, class_names=['Genuine', 'Fraud'])
        for ith_fraud in range(10): #range(len(frauds_indices_found_by_model)):
            print(ith_fraud)
            exp = explainer.explain_instance(x_test[frauds_indices_found_by_model[ith_fraud]], model.predict, num_features=10, top_labels=2)
            for elem in exp.local_exp[0]:
                to_show[feature_names[elem[0] % len(feature_names)]] += 1
    print(sorted(to_show.items(), key=lambda x_: abs(x_[1]), reverse=True))

def explain_sample(model, x_train, x_test):
    feature_names = get_features()
    if type(model) != Sequential:
        explainer = lime.lime_tabular.LimeTabularExplainer(x_train, feature_names=feature_names, class_names=["Genuine", "Fraud"], discretize_continuous=True)
        exp = explainer.explain_instance(x_test[0], model.predict_proba, num_features=10, top_labels=2)
        print("Features more important: ")
        print(exp.local_exp)
        for elem in exp.local_exp[0]:
            print(feature_names[elem[0]], abs(elem[1]))
    else:
        explainer = lime.lime_tabular.RecurrentTabularExplainer(x_train, feature_names=feature_names, discretize_continuous=True, class_names=['Genuine', 'Fraud'])
        exp = explainer.explain_instance(x_test[0], model.predict, num_features=10, top_labels=2)
        print("Features more important: ")
        print(exp.local_exp)
        for elem in exp.local_exp[0]:
            print(feature_names[elem[0] % len(feature_names)], abs(elem[1]))

# given a set, it prints the number of times each features has been in the top 5 features
def explain_dataset(model, x_train, x_test, threshold):
    if type(model) != Sequential:
        y_pred = model.predict_proba(x_test)
        y_pred = y_pred[:, 1]
    else:
        y_pred = model.predict(x_test)
        y_pred = y_pred.ravel()

    y_pred = evaluation.adjusted_classes(y_pred, threshold)
    if x_test.shape[0] != 1:
        frauds_indices_found_by_model = np.where(np.array(y_pred) == 1)
        frauds_indices_found_by_model = np.array(frauds_indices_found_by_model).ravel()
        explain_frauds(model, x_train, x_test, frauds_indices_found_by_model)
    else:
        print("The model output: ", y_pred[0])
        explain_sample(model, x_train, x_test)