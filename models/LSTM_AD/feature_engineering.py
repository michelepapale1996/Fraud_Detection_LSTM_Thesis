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
def feature_engineering(user_transactions):
    # user_transactions = user_transactions.iloc[0:100]
    len_user_transactions = len(user_transactions.index)
    user_transactions.loc[:, "count_trx_iban"] = np.zeros(len_user_transactions)
    user_transactions.loc[:, "time_delta"] = np.zeros(len_user_transactions)
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
                if i == j + 1:
                    time_delta = user_transactions.iloc[i]["Timestamp"] - user_transactions.iloc[j]["Timestamp"]
                    time_delta = time_delta.total_seconds()

                if user_transactions.iloc[i]["IBAN"] == user_transactions.iloc[j]["IBAN"]:
                    count_trx_iban += 1

        user_transactions.loc[user_transactions.iloc[i:i + 1].index[0], "count_trx_iban"] = count_trx_iban
        user_transactions.loc[user_transactions.iloc[i:i + 1].index[0], "time_delta"] = time_delta
        user_transactions.loc[user_transactions.iloc[i:i + 1].index[0], "count_foreign_asn"] = count_foreign_asn
        user_transactions.loc[user_transactions.iloc[i:i + 1].index[0], "count_national_asn"] = count_national_asn
        user_transactions.loc[user_transactions.iloc[i:i + 1].index[0], "count_foreign_iban"] = count_foreign_iban
        user_transactions.loc[user_transactions.iloc[i:i + 1].index[0], "count_national_iban"] = count_national_iban
        cols = ["Importo",
                "count_trx_iban",
                "time_delta",
                "count_foreign_asn",
                "count_national_asn",
                "count_foreign_iban",
                "count_national_iban",
                "Timestamp",
                "isFraud"]
    return user_transactions[cols]

def get_number_user_transactions_per_day(user):
    def daterange(start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)

    start_date = user["Timestamp"].iloc[0]
    end_date = user["Timestamp"].iloc[len(user.index.values) - 1]
    transactions_per_day = []
    for single_date in daterange(start_date, end_date):
        num_transactions_in_day = 0
        for i in range(len(user["Timestamp"].index.values)):
            if user["Timestamp"].iloc[i].date() == single_date:
                # print("found: ", single_date.strftime("%Y-%m-%d"), ", ", trx_user["Timestamp"].iloc[i].date())
                num_transactions_in_day += 1
        transactions_per_day.append(num_transactions_in_day)
    return transactions_per_day