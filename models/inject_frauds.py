from datetime import timedelta, datetime
import numpy as np
import random
import pandas as pd
import time

# fix random seed for reproducibility
np.random.seed(7)

thirty_days = timedelta(30)
one_day = timedelta(1)
seven_days = timedelta(7)
two_weeks = timedelta(14)
one_hour = timedelta(0, 60 * 60)
num_frauds = 5

def random_date(previous, next):
    return datetime.fromtimestamp((next.timestamp() - previous.timestamp()) * random.random() + previous.timestamp())

def insert_row(idx, df, row):
    dfA = df.iloc[:idx, ]
    dfB = df.iloc[idx:, ]
    df = dfA.append(row).append(dfB).reset_index(drop = True)
    return df

'''
def first_scenario(dataset_val_test):
    for i in range(num_frauds):
        print("----------------------")
        index = random.choice(dataset_val_test.index[1:])
        importo = random.randint(100, 500)
        # let's assume that the iban does not belong to the set of iban where the user send money
        count_trx_iban = 1
        timestamp_previous_trx = pd.Timestamp(dataset_val_test.loc[index - 1, "Timestamp"])
        timestamp_next_trx = pd.Timestamp(dataset_val_test.loc[index, "Timestamp"])
        timestamp_fraud = random_date(timestamp_previous_trx, timestamp_next_trx)
        print(timestamp_previous_trx, " - ", timestamp_next_trx)
        print(timestamp_fraud)
        time_delta = timestamp_fraud.timestamp() - timestamp_previous_trx.timestamp()
        trx_in_one_day = 1
        trx_in_one_hour = 1
        trx_in_one_month = 1
        trx_in_two_weeks = 1
        trx_in_one_week = 1

        if index - 1 not in dataset_val_test.index:
            count_national_asn = 1
            count_national_iban = 1
            trx_in_one_day = 1
            trx_in_one_hour = 1
            trx_in_one_month = 1
            trx_in_two_weeks = 1
            trx_in_one_week = 1
        else:
            if time_delta < one_day.total_seconds():
                trx_in_one_day += 1

            if time_delta < thirty_days.total_seconds():
                trx_in_one_month += 1

            if time_delta < two_weeks.total_seconds():
                trx_in_two_weeks += 1

            if time_delta < seven_days.total_seconds():
                trx_in_one_week += 1

            if time_delta < one_hour.total_seconds():
                trx_in_one_hour += 1

            count_national_asn = dataset_val_test.loc[index - 1, "count_national_asn"] + 1
            count_national_iban = dataset_val_test.loc[index - 1, "count_national_iban"] + 1
        count_foreign_iban = 0
        count_foreign_asn = 0
        isFraud = 1
        new_row = [importo,
                   count_trx_iban,
                   time_delta,
                   trx_in_one_day,
                   trx_in_one_hour,
                   trx_in_one_month,
                   trx_in_two_weeks,
                   trx_in_one_week,
                   count_foreign_asn,
                   count_national_asn,
                   count_foreign_iban,
                   count_national_iban,
                   timestamp_fraud,
                   isFraud]
        cols = list(dataset_val_test.columns)
        dataset_val_test = insert_row(index, dataset_val_test, pd.DataFrame([new_row], columns=cols))
    return dataset_val_test
'''

def second_scenario(dataset_val_test):
    for i in range(num_frauds):
        index = random.choice(dataset_val_test.index[1:])
        timestamp_previous_trx = pd.Timestamp(dataset_val_test.loc[index - 1, "Timestamp"])
        timestamp_next_trx = pd.Timestamp(dataset_val_test.loc[index, "Timestamp"])
        timestamp_fraud = random_date(timestamp_previous_trx, timestamp_next_trx)
        importo = random.randint(500, 1500)
        # let's assume that the iban does not belong to the set of iban where the user send money
        count_trx_iban = 1
        time_delta = timestamp_fraud.timestamp() - timestamp_previous_trx.timestamp()
        trx_in_one_day = 1
        trx_in_one_hour = 1
        trx_in_one_month = 1
        trx_in_two_weeks = 1
        trx_in_one_week = 1

        if index - 1 not in dataset_val_test.index:
            count_national_asn = 1
            count_national_iban = 1
            trx_in_one_day = 1
            trx_in_one_hour = 1
            trx_in_one_month = 1
            trx_in_two_weeks = 1
            trx_in_one_week = 1
        else:
            if time_delta < one_day.total_seconds():
                trx_in_one_day += 1

            if time_delta < thirty_days.total_seconds():
                trx_in_one_month += 1

            if time_delta < two_weeks.total_seconds():
                trx_in_two_weeks += 1

            if time_delta < seven_days.total_seconds():
                trx_in_one_week += 1

            if time_delta < one_hour.total_seconds():
                trx_in_one_hour += 1

            count_national_asn = dataset_val_test.loc[index - 1, "count_national_asn"] + 1
            count_national_iban = dataset_val_test.loc[index - 1, "count_national_iban"] + 1
        count_foreign_iban = 0
        count_foreign_asn = 0
        isFraud = 1
        new_row = [importo,
                   count_trx_iban,
                   time_delta,
                   trx_in_one_day,
                   trx_in_one_hour,
                   trx_in_one_month,
                   trx_in_two_weeks,
                   trx_in_one_week,
                   count_foreign_asn,
                   count_national_asn,
                   count_foreign_iban,
                   count_national_iban,
                   timestamp_fraud,
                   isFraud]
        cols = list(dataset_val_test.columns)
        dataset_val_test = insert_row(index, dataset_val_test, pd.DataFrame([new_row], columns=cols))
    return dataset_val_test

def third_scenario(dataset_val_test):
    for i in range(num_frauds):
        index = random.choice(dataset_val_test.index)
        timestamp_previous_trx = pd.Timestamp(dataset_val_test.loc[index - 1, "Timestamp"])
        timestamp_next_trx = pd.Timestamp(dataset_val_test.loc[index, "Timestamp"])
        timestamp_fraud = random_date(timestamp_previous_trx, timestamp_next_trx)
        importo = random.randint(1500, 3000)
        # let's assume that the iban does not belong to the set of iban where the user send money
        count_trx_iban = 1
        time_delta = timestamp_fraud.timestamp() - timestamp_previous_trx.timestamp()
        trx_in_one_day = 1
        trx_in_one_hour = 1
        trx_in_one_month = 1
        trx_in_two_weeks = 1
        trx_in_one_week = 1

        if index - 1 not in dataset_val_test.index:
            count_national_asn = 1
            count_national_iban = 1
            trx_in_one_day = 1
            trx_in_one_hour = 1
            trx_in_one_month = 1
            trx_in_two_weeks = 1
            trx_in_one_week = 1
        else:
            if time_delta < one_day.total_seconds():
                trx_in_one_day += 1

            if time_delta < thirty_days.total_seconds():
                trx_in_one_month += 1

            if time_delta < two_weeks.total_seconds():
                trx_in_two_weeks += 1

            if time_delta < seven_days.total_seconds():
                trx_in_one_week += 1

            if time_delta < one_hour.total_seconds():
                trx_in_one_hour += 1

            count_national_asn = dataset_val_test.loc[index - 1, "count_national_asn"] + 1
            count_national_iban = dataset_val_test.loc[index - 1, "count_national_iban"] + 1
        count_foreign_iban = 0
        count_foreign_asn = 0
        isFraud = 1
        new_row = [importo,
                   count_trx_iban,
                   time_delta,
                   trx_in_one_day,
                   trx_in_one_hour,
                   trx_in_one_month,
                   trx_in_two_weeks,
                   trx_in_one_week,
                   count_foreign_asn,
                   count_national_asn,
                   count_foreign_iban,
                   count_national_iban,
                   timestamp_fraud,
                   isFraud]
        cols = list(dataset_val_test.columns)
        dataset_val_test = insert_row(index, dataset_val_test, pd.DataFrame([new_row], columns=cols))
    return dataset_val_test

def first_scenario(user_dataset, real_dataset, last_index_train, last_index_val):
    def craft_fraud(user_dataset):
        ip = real_dataset.loc[random.choice(real_dataset.index), "IP"]
        idSessione = real_dataset.loc[random.choice(real_dataset.index), "IDSessione"]
        timestamp_previous_trx = pd.Timestamp(user_dataset.loc[index - 1, "Timestamp"])
        timestamp_next_trx = pd.Timestamp(user_dataset.loc[index, "Timestamp"])
        timestamp_fraud = random_date(timestamp_previous_trx, timestamp_next_trx)
        importo = random.randint(100, 500)
        msgErrore = real_dataset.loc[random.choice(real_dataset.index), "MsgErrore"]
        userID = user_dataset.loc[index, "UserID"]
        iban = real_dataset.loc[random.choice(real_dataset.index), "IBAN"]
        numConfermaSMS = real_dataset.loc[random.choice(real_dataset.index), "NumConfermaSMS"]
        iban_cc = "IT"
        cc_asn = user_dataset.loc[index, "CC_ASN"]
        isFraud = 1
        new_row = [ip,
                   idSessione,
                   timestamp_fraud,
                   importo,
                   msgErrore,
                   userID,
                   iban,
                   numConfermaSMS,
                   iban_cc,
                   cc_asn,
                   isFraud]
        cols = list(user_dataset.columns)
        return pd.DataFrame([new_row], columns=cols)

    for i in range(num_frauds):
        index = random.choice(user_dataset.index[last_index_train:last_index_val])
        new_fraud = craft_fraud(user_dataset)
        user_dataset = insert_row(index, user_dataset, new_fraud)

        index = random.choice(user_dataset.index[last_index_val:])
        new_fraud = craft_fraud(user_dataset)
        user_dataset = insert_row(index, user_dataset, new_fraud)

    return user_dataset