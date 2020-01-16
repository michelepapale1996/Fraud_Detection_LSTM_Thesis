import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, minmax_scale
import seaborn as sns
import matplotlib.pyplot as plt
# seaborn can generate several warnings, we ignore them
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("/home/mpapale/thesis")
from dataset_creation import inject_frauds, constants
from sklearn.preprocessing import MinMaxScaler
from dataset_creation.constants import *
import random
import math

# in order to print all the columns
pd.set_option('display.max_columns', 100)
sns.set(style="white", color_codes=True)
sns.set_context(rc={"font.family":'sans',"font.size":24,"axes.titlesize":24,"axes.labelsize":24})

# -----------------------------------------------------------------------------------------------
# -------------------------- Feature engineering module -----------------------------------------
# -----------------------------------------------------------------------------------------------

def read_dataset():
    # reading the datasets
    bonifici = pd.read_csv("../datasets/quiubi_bonifici.csv")
    segnalaz = pd.read_csv("../datasets/bonifici_segnalaz.csv")
    bonifici.set_index('indice', inplace=True)
    segnalaz.set_index('indice', inplace=True)

    # dropping columns with useless data
    useless_features = ["IDSessione", "CAP", "Servizio", "Status", "Paese", "Provincia", "Nazione", "IDTransazione",
                        "CRO", "Causale", "Valuta", "ProfSicurezza", "NumConto", "ABI", "CAB", "Intestatario",
                        "Indirizzo"]
    bonifici = bonifici.drop(useless_features, axis=1)
    segnalaz = segnalaz.drop(useless_features, axis=1)
    # in future, try to use these features
    bonifici = bonifici.drop(["DataValuta", "DataEsecuzione", "Nominativo", "TipoOperazione"], axis=1)
    segnalaz = segnalaz.drop(["DataValuta", "DataEsecuzione", "Nominativo", "TipoOperazione"], axis=1)

    # datasets merge into bonifici
    bonifici["isFraud"] = np.zeros(len(bonifici.index))
    for index, row in segnalaz.iterrows():
        if index in bonifici.index:
            bonifici.loc[index, "isFraud"] = 1
        else:
            bonifici.append(row)
    bonifici["isFraud"] = pd.to_numeric(bonifici["isFraud"], downcast='integer')
    bonifici.Timestamp = pd.to_datetime(bonifici.Timestamp)
    return bonifici

def read_dataset_fraud_buster():
    # reading the datasets
    bonifici = pd.read_csv("../datasets/fraud_buster_users_min_100_transactions.csv", parse_dates=True, sep=",")

    bonifici.Timestamp = pd.to_datetime(bonifici.Timestamp)
    return bonifici

def to_boolean(dataset):
    print("Changing NumConfermaSMS to boolean...")
    dataset.NumConfermaSMS = dataset.NumConfermaSMS.eq('Si').astype(int)
    print("Changing MsgErrore to boolean...")
    # MsgErrore changed to boolean
    dataset.MsgErrore.fillna(0, inplace=True)
    dataset["MsgErrore"] = dataset["MsgErrore"].apply(lambda x: 1 if x != 0 else 0)
    return dataset

def create_engineered_features(dataset):
    print("Creating isItalianSender and isItalianReceiver...")
    # creating "isItalianSender", "isItalianReceiver"
    dataset["isItalianSender"] = np.ones(len(dataset.index))
    dataset["isItalianReceiver"] = np.ones(len(dataset.index))

    for index, row in dataset[["CC_ASN", "IBAN_CC"]].iterrows():
        if row["CC_ASN"][:2] != "IT":
            dataset.at[index, "isItalianSender"] = 0
        if row["IBAN_CC"] != "IT":
            dataset.at[index, "isItalianReceiver"] = 0
    dataset["isItalianSender"] = pd.to_numeric(dataset["isItalianSender"], downcast='integer')
    dataset["isItalianReceiver"] = pd.to_numeric(dataset["isItalianReceiver"], downcast='integer')

    return dataset

# get transactions of a given user and returns the user transactions with new, aggregated, features
def create_user_aggregated_features(user_transactions):
    user_transactions = user_transactions.sort_values("Timestamp").reset_index(drop=True)

    for i in range(len(user_transactions)):
        if i == 0:
            # possible ways to handle timedelta of first transactions:
            # set it equal to timedelta of second transaction
            # time_delta = user_transactions.iloc[i + 1]["Timestamp"] - user_transactions.iloc[i]["Timestamp"]
            # or set it random in seven days
            time_delta = timedelta(random.uniform(0, 7))
        else:
            time_delta = user_transactions.iloc[i]["Timestamp"] - user_transactions.iloc[i - 1]["Timestamp"]
        time_delta = time_delta.total_seconds()
        user_transactions.at[i, "time_delta"] = time_delta

    '''
    # time window: one week, one month and global
    # 1000 is used because in the dataset there are less than 1000 days in dataset -> get all past transactions
    for time_window in [7, 30, 1000]:
        already_seen_ibans = {}
        for i in range(len(user_transactions)):
            window = user_transactions.iloc[:i]
            # when creating aggregated features, do not consider frauds
            window = window[window.isFraud == 0]
            window = window[window.Timestamp > user_transactions.iloc[i].Timestamp - timedelta(time_window)]
            count_trx = len(window.Importo)

            # if the current iteration involves the first transaction of the user
            if i == 0:
                w1 = user_transactions.iloc[:i + 1]
                w1 = w1[w1.isFraud == 0]
                w1 = w1[w1.Timestamp > user_transactions.iloc[i].Timestamp - timedelta(time_window)]
                mean_amount = w1.Importo.mean()

                w1 = user_transactions.iloc[:i + 2]
                w1 = w1[w1.isFraud == 0]
                w1 = w1[w1.Timestamp > user_transactions.iloc[i].Timestamp - timedelta(time_window)]
                stdev_amount = w1.Importo.std()

            # if the window is empty
            if count_trx == 0:
                mean_amount = 0
                stdev_amount = 0

            if math.isnan(mean_amount) or math.isnan(stdev_amount):
                print("ERRORE: ", i)
                raise RuntimeError("nan in mean or stdev when creating aggregated features. Error generated due to " + str(i) + "transaction for user " + str(user_transactions.iloc[0].UserID))
            count_different_iban_ccs = len(np.unique(window.IBAN_CC.values))
            count_different_asn_ccs = len(np.unique(window.CC_ASN.values))

            if user_transactions.iloc[i]["IBAN_CC"] in window.IBAN_CC.values:
                is_new_iban_cc = 0
            else:
                is_new_iban_cc = 1

            if user_transactions.iloc[i]["CC_ASN"] in window.CC_ASN.values:
                is_new_asn_cc = 0
            else:
                is_new_asn_cc = 1

            if user_transactions.iloc[i]["IBAN"] in window.IBAN.values:
                count_trx_iban = already_seen_ibans[user_transactions.iloc[i]["IBAN"]]
                already_seen_ibans[user_transactions.iloc[i]["IBAN"]] += 1
                is_new_iban = 0
            else:
                already_seen_ibans[user_transactions.iloc[i]["IBAN"]] = 1
                is_new_iban = 1
                count_trx_iban = 0

            if user_transactions.iloc[i]["IP"] in window.IP.values:
                is_new_ip = 0
            else:
                is_new_ip = 1

            # print(count_different_iban_ccs, count_different_asn_ccs, count_trx, mean_amount, stdev_amount, count_trx_iban)
            user_transactions.at[i, "count_different_iban_ccs_" + str(time_window) + "_window"] = count_different_iban_ccs
            user_transactions.at[i, "count_different_asn_ccs_" + str(time_window) + "_window"] = count_different_asn_ccs
            user_transactions.at[i, "count_trx_" + str(time_window) + "_window"] = count_trx
            user_transactions.at[i, "mean_amount_" + str(time_window) + "_window"] = mean_amount
            user_transactions.at[i, "stdev_amount_" + str(time_window) + "_window"] = stdev_amount
            user_transactions.at[i, "count_trx_iban_" + str(time_window) + "_window"] = count_trx_iban
            user_transactions.at[i, "is_new_asn_cc_" + str(time_window) + "_window"] = is_new_asn_cc
            user_transactions.at[i, "is_new_iban_" + str(time_window) + "_window"] = is_new_iban
            user_transactions.at[i, "is_new_iban_cc_" + str(time_window) + "_window"] = is_new_iban_cc
            user_transactions.at[i, "is_new_ip_" + str(time_window) + "_window"] = is_new_ip
    '''
    return user_transactions

def create_aggregated_features(dataset):
    print("Creating aggregated features...")
    dataset_by_user = dataset.groupby("UserID")
    counter = 0
    for user in dataset_by_user.groups.keys():
        # print(counter, ") user: ", user)
        counter += 1
        group = dataset_by_user.get_group(user).sort_values(by='Timestamp', ascending=True).reset_index(drop=True)
        group_with_aggregated_features = create_user_aggregated_features(group)
        try:
            dataset_with_aggregated_features = dataset_with_aggregated_features.append(group_with_aggregated_features, ignore_index=True)
        # if it is the first iteration, dataset_with_aggregated_features is not defined
        except NameError:
            dataset_with_aggregated_features = group_with_aggregated_features

    return dataset_with_aggregated_features

def scale_features(train, test):
    scaler = MinMaxScaler()

    df_dtypes = train.dtypes.values.tolist()
    selected_dtypes = [np.dtype('int64'), np.dtype('int8'), np.dtype('float64')]
    selected_cols = [True if x in selected_dtypes else False for x in df_dtypes]
    col_names = train.loc[:, selected_cols].columns

    train_features_to_scale = train.loc[:, selected_cols]
    test_features_to_scale = test.loc[:, selected_cols]

    train_rescaled = scaler.fit_transform(train_features_to_scale.values)
    test_rescaled = scaler.transform(test_features_to_scale.values)
    # converting in pandas dataframe
    train_rescaled = pd.DataFrame(train_rescaled, columns=col_names).reset_index(drop=True)
    test_rescaled = pd.DataFrame(test_rescaled, columns=col_names).reset_index(drop=True)

    train = train.drop(col_names, axis=1).reset_index(drop=True)
    test = test.drop(col_names, axis=1).reset_index(drop=True)

    train = train.join(train_rescaled)
    test = test.join(test_rescaled)
    return train, test


if __name__ == "__main__":
    if constants.DATASET_TYPE == constants.INJECTED_DATASET:
        complete_dataset = read_dataset()
        # N.B. CONSIDER ONLY GENUINE TRANSACTIONS
        dataset = complete_dataset[complete_dataset.isFraud == 0]
        dataset = dataset.reset_index(drop=True)
        users = constants.users_with_more_than_50_trx_train_5_trx_test
        print("In total, there are ", len(users), "users")
        # considering only users with more transactions
        dataset = dataset[dataset.UserID.isin(users)]

    if constants.DATASET_TYPE == constants.FRAUD_BUSTER_DATASET:
        dataset = read_dataset_fraud_buster()
        # this dataset does not contain any fraud known
        dataset["isFraud"] = np.zeros(len(dataset.UserID.values))
        dataset = dataset.drop(["prova", "TipoOperazione"], axis=1)
        print("There are", len(dataset.UserID.value_counts()), "users.")

    if constants.DATASET_TYPE == constants.REAL_DATASET:
        complete_dataset = read_dataset()
        dataset = complete_dataset.reset_index(drop=True)
        users = constants.users_with_more_than_10_trx_train_5_trx_test
        dataset = dataset[dataset.UserID.isin(users)]

    thirty_days = timedelta(30)
    first_date = min(dataset.Timestamp)
    last_date = max(dataset.Timestamp)
    last_date_train_set = last_date - thirty_days
    scaler = MinMaxScaler()
    dataset_by_user = dataset.groupby("UserID")

    if constants.DATASET_TYPE == constants.INJECTED_DATASET or constants.DATASET_TYPE == constants.FRAUD_BUSTER_DATASET:
        # divide train/test because must inject frauds both in train and in test
        d_train = dataset[dataset.Timestamp < last_date_train_set]
        d_test = dataset[dataset.Timestamp >= last_date_train_set]

        scenarios = [FIRST_SCENARIO, SECOND_SCENARIO, THIRD_SCENARIO, FOURTH_SCENARIO, FIFTH_SCENARIO, SIXTH_SCENARIO, SEVENTH_SCENARIO, EIGHTH_SCENARIO, NINTH_SCENARIO, ALL_SCENARIOS]
        # scenarios = [FIRST_SCENARIO, SECOND_SCENARIO, THIRD_SCENARIO]
        # scenarios = [FOURTH_SCENARIO, FIFTH_SCENARIO, SIXTH_SCENARIO]
        # scenarios = [SEVENTH_SCENARIO, EIGHTH_SCENARIO, NINTH_SCENARIO, ALL_SCENARIOS]
        for scenario_type in scenarios:
            print("New scenario: ", scenario_type)
            dataset_train = inject_frauds.craft_frauds(d_train, dataset, scenario_type, last_date_train_set)
            dataset_test = inject_frauds.craft_frauds(d_test, dataset, scenario_type, last_date_train_set)

            d = dataset_train.append(dataset_test, ignore_index=True)

            d = to_boolean(d)
            d = create_engineered_features(d)
            dataset_with_aggregated_features = create_aggregated_features(d)

            dataset_train_with_aggregated_features = dataset_with_aggregated_features[dataset_with_aggregated_features.Timestamp < last_date_train_set]
            dataset_test_with_aggregated_features = dataset_with_aggregated_features[dataset_with_aggregated_features.Timestamp >= last_date_train_set]

            dataset_train_with_aggregated_features, dataset_test_with_aggregated_features = scale_features(dataset_train_with_aggregated_features, dataset_test_with_aggregated_features)

            # saving the dataset
            if constants.DATASET_TYPE == constants.INJECTED_DATASET:
                dataset_train_with_aggregated_features.to_csv("../datasets/train_" + str(len(dataset.UserID.value_counts())) + "_users_" + scenario_type + "_scenario.csv", index=False)
                dataset_test_with_aggregated_features.to_csv("../datasets/test_" + str(len(dataset.UserID.value_counts())) + "_users_" + scenario_type + "_scenario.csv", index=False)
            else:
                dataset_train_with_aggregated_features.to_csv("../datasets/fraud_buster_train_" + str(len(dataset.UserID.value_counts())) + "_users_" + scenario_type + "_scenario.csv", index=False)
                dataset_test_with_aggregated_features.to_csv("../datasets/fraud_buster_test_" + str(len(dataset.UserID.value_counts())) + "_users_" + scenario_type + "_scenario.csv", index=False)

    elif constants.DATASET_TYPE == constants.REAL_DATASET:
        # in real dataset there is not the injection of the frauds
        dataset = to_boolean(dataset)
        dataset = create_engineered_features(dataset)
        dataset_with_aggregated_features = create_aggregated_features(dataset)

        dataset_train_with_aggregated_features = dataset_with_aggregated_features[dataset_with_aggregated_features.Timestamp < last_date_train_set]
        dataset_test_with_aggregated_features = dataset_with_aggregated_features[dataset_with_aggregated_features.Timestamp >= last_date_train_set]

        dataset_train_with_aggregated_features, dataset_test_with_aggregated_features = scale_features(dataset_train_with_aggregated_features, dataset_test_with_aggregated_features)

        dataset_train_with_aggregated_features.to_csv("../datasets/real_dataset_train_" + str(len(dataset.UserID.value_counts())) + "_users.csv", index=False)
        dataset_test_with_aggregated_features.to_csv("../datasets/real_dataset_test_" + str(len(dataset.UserID.value_counts())) + "_users.csv",index=False)