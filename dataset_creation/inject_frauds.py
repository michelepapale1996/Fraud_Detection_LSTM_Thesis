from datetime import timedelta, datetime
import numpy as np
import random
import pandas as pd
import time
from dataset_creation.constants import *

# fix random seed for reproducibility
np.random.seed(7)

thirty_days = timedelta(30)
one_day = timedelta(1)
seven_days = timedelta(7)
two_weeks = timedelta(14)
one_hour = timedelta(0, 60 * 60)
one_minute = timedelta(0, 60)

def random_date(previous, next):
    return datetime.fromtimestamp((next.timestamp() - previous.timestamp()) * random.random() + previous.timestamp())

def insert_row(idx, df, row):
    dfA = df.iloc[:idx, ]
    dfB = df.iloc[idx:, ]
    df = dfA.append(row).append(dfB).reset_index(drop=True)
    return df

def first_scenario():
    importo = random.randint(100, 500)
    iban_cc = "IT"
    isFraud = 1
    cc_asn = "IT000"
    return importo, iban_cc, cc_asn, isFraud

def second_scenario():
    importo = random.randint(500, 1500)
    iban_cc = "IT"
    isFraud = 1
    cc_asn = "IT000"
    return importo, iban_cc, cc_asn, isFraud

def third_scenario():
    importo = random.randint(1500, 3000)
    iban_cc = "IT"
    isFraud = 1
    cc_asn = "IT000"
    return importo, iban_cc, cc_asn, isFraud

# always same iban receiver
def fourth_scenario():
    importo = random.randint(500, 1500)
    iban_cc = "IT"
    isFraud = 1
    cc_asn = "IT000"
    return importo, iban_cc, cc_asn, isFraud

def fifth_scenario():
    importo = random.randint(500, 1500)
    iban_cc = "IT"
    isFraud = 1
    cc_asn = "IT000"
    return importo, iban_cc, cc_asn, isFraud

def sixth_scenario():
    importo = random.randint(500, 1500)
    iban_cc = "IT"
    isFraud = 1
    cc_asn = "IT000"
    return importo, iban_cc, cc_asn, isFraud

# foreign iban receiver
def seventh_scenario():
    importo = random.randint(500, 1500)
    iban_cc = "RM"
    isFraud = 1
    cc_asn = "IT000"
    return importo, iban_cc, cc_asn, isFraud

# foreign asn
def eighth_scenario():
    importo = random.randint(500, 1500)
    iban_cc = "IT"
    isFraud = 1
    cc_asn = "RM000"
    return importo, iban_cc, cc_asn, isFraud

def ninth_scenario():
    importo = random.randint(2500, 7500)
    iban_cc = "IT"
    isFraud = 1
    cc_asn = "IT000"
    return importo, iban_cc, cc_asn, isFraud

# dispatcher is used to call relative functions
dispatcher = {
    FIRST_SCENARIO: first_scenario,
    SECOND_SCENARIO: second_scenario,
    THIRD_SCENARIO: third_scenario,
    FOURTH_SCENARIO: fourth_scenario,
    FIFTH_SCENARIO: fifth_scenario,
    SIXTH_SCENARIO: sixth_scenario,
    SEVENTH_SCENARIO: seventh_scenario,
    EIGHTH_SCENARIO: eighth_scenario,
    NINTH_SCENARIO: ninth_scenario
}

# Following the most interesting fields:
# - Amount
# - Beneficiary distribution (equally, all to one, max one)
# - IBAN (national, foreign)
# - ASN (national, foreign)
def craft_frauds(dataset, real_dataset, scenario_type, last_date_train_set, all_in_same_day=True):
    if scenario_type == ALL_SCENARIOS:
        scenario_types = [FIRST_SCENARIO, SECOND_SCENARIO, THIRD_SCENARIO, FOURTH_SCENARIO, FIFTH_SCENARIO, SIXTH_SCENARIO, SEVENTH_SCENARIO, EIGHTH_SCENARIO, NINTH_SCENARIO]
    else:
        scenario_types = [scenario_type]
    dataset_by_user = dataset.groupby("UserID")
    real_dataset_by_user = real_dataset.groupby("UserID")
    all_users = list(real_dataset_by_user.groups.keys())

    users = dataset_by_user.groups.keys()
    num_users_to_infect = int(len(users) / 100 * 5)
    print(num_users_to_infect, "users will be infected.")
    for type in scenario_types:
        print("Injecting frauds for this scenario...")
        users_to_infect = random.sample(users, num_users_to_infect)

        if type == FIRST_SCENARIO or type == SECOND_SCENARIO or type == THIRD_SCENARIO or type == FOURTH_SCENARIO or type == FIFTH_SCENARIO:
            num_frauds = 10
        else:
            num_frauds = 5

        for user in users_to_infect:
            # print("Injecting frauds in user", user)
            user_dataset = dataset_by_user.get_group(user).sort_values(by="Timestamp").reset_index(drop=True)

            index = random.choice(user_dataset.index)
            while index - 1 not in user_dataset.index:
                index = random.choice(user_dataset.index)

            # injecting frauds in set
            for i in range(num_frauds):
                if all_in_same_day:
                    new_fraud, all_users = craft_single_fraud(user_dataset, real_dataset, index, type, all_users, last_date_train_set, time_lapse=one_hour)
                else:
                    new_fraud, all_users = craft_single_fraud(user_dataset, real_dataset, index, type, all_users, last_date_train_set, time_lapse=thirty_days)
                user_dataset = insert_row(index, user_dataset, new_fraud)
                dataset = dataset.append(new_fraud)
                index += 1
    print("Num of frauds in dataset", len(dataset[dataset.isFraud == 1]))
    return dataset

# craft a single fraud.
# all_users is used in that scenarios where the iban receiver must not be the same/must be different for each fraud crafted
# time_lapse is the interval in which all the frauds must be injected
def craft_single_fraud(user_dataset, real_dataset, index, fraud_type, all_users, last_date_train_set, time_lapse=False):
    # resetting the index. In this way it's simpler to get the previous transaction
    user_dataset = user_dataset.reset_index(drop=True)
    ip = real_dataset.loc[random.choice(real_dataset.index), "IP"]
    timestamp_previous_trx = pd.Timestamp(user_dataset.loc[index - 1, "Timestamp"])

    if time_lapse == False:
        timestamp_next_trx = pd.Timestamp(user_dataset.loc[index, "Timestamp"])
    else:
        timestamp_next_trx = random.uniform(timestamp_previous_trx + one_minute, timestamp_previous_trx + time_lapse)

    # timestamp_fraud = random_date(timestamp_previous_trx, timestamp_next_trx)
    timestamp_fraud = timestamp_next_trx

    msgErrore = real_dataset.loc[random.choice(real_dataset.index), "MsgErrore"]
    userID = user_dataset.loc[index, "UserID"]
    tipoOperazione = user_dataset.loc[index, "TipoOperazione"]
    numConfermaSMS = real_dataset.loc[random.choice(real_dataset.index), "NumConfermaSMS"]
    asn = real_dataset.loc[random.choice(real_dataset.index), "CC_ASN"][2:]

    if fraud_type == FIRST_SCENARIO or fraud_type == SECOND_SCENARIO or fraud_type == THIRD_SCENARIO or fraud_type == SEVENTH_SCENARIO or fraud_type == EIGHTH_SCENARIO or fraud_type == NINTH_SCENARIO:
        iban = real_dataset.loc[random.choice(real_dataset.index), "IBAN"]
    elif fraud_type == FOURTH_SCENARIO:
        # all frauds towards the same iban
        iban = real_dataset.iloc[0]["IBAN"]
    else:
        # all frauds towards different iban
        iban = all_users[0]
        all_users = all_users[1:]

    importo, iban_cc, cc_asn, isFraud = dispatcher[fraud_type]()

    if DATASET_TYPE != FRAUD_BUSTER_DATASET:
        new_row = [ip, timestamp_fraud, tipoOperazione, importo, msgErrore, userID, iban, numConfermaSMS, iban_cc, cc_asn, isFraud]
    else:
        new_row = [timestamp_fraud, importo, userID, iban, iban_cc, cc_asn, asn, ip, isFraud]

    cols = list(user_dataset.columns)
    return pd.DataFrame([new_row], columns=cols), all_users