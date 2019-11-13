import numpy as np
from datetime import timedelta, datetime
# fix random seed for reproducibility
np.random.seed(7)

thirty_days = timedelta(30)
one_day = timedelta(1)
seven_days = timedelta(7)
two_weeks = timedelta(14)
one_hour = timedelta(0, 60 * 60)

# todo: test created features. create trx_in_working_hours
# NB: TRX_IN_ONE_HOUR indica il numero di transazioni fatte nell'ultima ora prima di questa transazione
def feature_engineering(user_transactions):
    # user_transactions = user_transactions.iloc[0:100]
    len_user_transactions = len(user_transactions.index)
    user_transactions.loc[:, "count_trx_iban"] = np.zeros(len_user_transactions)
    user_transactions.loc[:, "time_delta"] = np.zeros(len_user_transactions)
    user_transactions.loc[:, "trx_in_one_month"] = np.zeros(len_user_transactions)
    user_transactions.loc[:, "trx_in_one_day"] = np.zeros(len_user_transactions)
    user_transactions.loc[:, "trx_in_two_weeks"] = np.zeros(len_user_transactions)
    user_transactions.loc[:, "trx_in_one_week"] = np.zeros(len_user_transactions)
    user_transactions.loc[:, "trx_in_one_hour"] = np.zeros(len_user_transactions)
    user_transactions.loc[:, "count_national_iban"] = np.zeros(len_user_transactions)
    user_transactions.loc[:, "count_foreign_iban"] = np.zeros(len_user_transactions)
    user_transactions.loc[:, "count_national_asn"] = np.zeros(len_user_transactions)
    user_transactions.loc[:, "count_foreign_asn"] = np.zeros(len_user_transactions)
    for i in range(len_user_transactions):
        count_trx_iban = 0
        time_delta = 0
        count_national_iban = 0
        count_foreign_iban = 0
        count_national_asn = 0
        count_foreign_asn = 0
        trx_in_one_day = 1
        trx_in_one_month = 1
        trx_in_two_weeks = 1
        trx_in_one_week = 1
        trx_in_one_hour = 1

        if user_transactions.iloc[i]["CC_ASN"][:2] != "IT":
            count_foreign_asn += 1
        else:
            count_national_asn += 1

        if user_transactions.iloc[i]["IBAN_CC"] != "IT":
            count_foreign_iban += 1
        else:
            count_national_iban += 1

        for j in range(0, i):
            if user_transactions.iloc[j]["isFraud"] == 0:
                delta = (user_transactions.iloc[i]["Timestamp"] - user_transactions.iloc[j]["Timestamp"]).total_seconds()
                if delta < one_day.total_seconds():
                    trx_in_one_day += 1

                if delta < thirty_days.total_seconds():
                    trx_in_one_month += 1

                if delta < two_weeks.total_seconds():
                    trx_in_two_weeks += 1

                if delta < seven_days.total_seconds():
                    trx_in_one_week += 1

                if delta < one_hour.total_seconds():
                    trx_in_one_hour += 1

                if i == j + 1:
                    time_delta = user_transactions.iloc[i]["Timestamp"] - user_transactions.iloc[j]["Timestamp"]
                    time_delta = time_delta.total_seconds()

                if user_transactions.iloc[i]["IBAN"] == user_transactions.iloc[j]["IBAN"]:
                    count_trx_iban += 1

        user_transactions.loc[user_transactions.iloc[i:i + 1].index[0], "count_trx_iban"] = count_trx_iban
        user_transactions.loc[user_transactions.iloc[i:i + 1].index[0], "time_delta"] = time_delta
        user_transactions.loc[user_transactions.iloc[i:i + 1].index[0], "trx_in_one_day"] = trx_in_one_day
        user_transactions.loc[user_transactions.iloc[i:i + 1].index[0], "trx_in_one_month"] = trx_in_one_month
        user_transactions.loc[user_transactions.iloc[i:i + 1].index[0], "trx_in_two_weeks"] = trx_in_two_weeks
        user_transactions.loc[user_transactions.iloc[i:i + 1].index[0], "trx_in_one_week"] = trx_in_one_week
        user_transactions.loc[user_transactions.iloc[i:i + 1].index[0], "trx_in_one_hour"] = trx_in_one_hour
        user_transactions.loc[user_transactions.iloc[i:i + 1].index[0], "count_foreign_asn"] = count_foreign_asn
        user_transactions.loc[user_transactions.iloc[i:i + 1].index[0], "count_national_asn"] = count_national_asn
        user_transactions.loc[user_transactions.iloc[i:i + 1].index[0], "count_foreign_iban"] = count_foreign_iban
        user_transactions.loc[user_transactions.iloc[i:i + 1].index[0], "count_national_iban"] = count_national_iban
        cols = ["Importo",
                "count_trx_iban",
                "time_delta",
                "trx_in_one_day",
                "trx_in_one_hour",
                "trx_in_one_month",
                "trx_in_two_weeks",
                "trx_in_one_week",
                "count_foreign_asn",
                "count_national_asn",
                "count_foreign_iban",
                "count_national_iban",
                "Timestamp",
                "isFraud"]
    return user_transactions[cols]