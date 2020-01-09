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
from dataset_creation import inject_frauds
import random

# in order to print all the columns
pd.set_option('display.max_columns', 100)
sns.set(style="white", color_codes=True)
sns.set_context(rc={"font.family": 'sans', "font.size": 24, "axes.titlesize": 24, "axes.labelsize": 24})


# -----------------------------------------------------------------------------------------------
# -------------------------- Feature engineering module -----------------------------------------
# -----------------------------------------------------------------------------------------------

def read_dataset():
    # reading the datasets
    bonifici = pd.read_csv("/home/mpapale/thesis/datasets/quiubi_bonifici.csv")
    bonifici.Timestamp = pd.to_datetime(bonifici.Timestamp)
    bonifici = bonifici.drop(
        ["IDSessione", "CAP", "Status", "Paese", "Provincia", "Nazione", "IDTransazione", "CRO", "Causale", "Valuta",
         "ProfSicurezza", "NumConto", "ABI", "CAB", "Intestatario", "Indirizzo"], axis=1)
    bonifici = bonifici.drop(["DataValuta", "DataEsecuzione", "Nominativo"], axis=1)
    bonifici.set_index('index', inplace=True)

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


def get_features_single_trx(group, ith_transaction):
    group.Timestamp = pd.to_datetime(group.Timestamp)
    count_trx_iban = 0
    is_new_asn_cc = 1
    is_new_iban = 1
    is_new_iban_cc = 1
    is_new_ip = 1
    time_delta = 0

    for j in range(0, ith_transaction):
        if group.iloc[j]["isFraud"] == 0:
            if ith_transaction == j + 1:
                time_delta = group.iloc[ith_transaction]["Timestamp"] - group.iloc[j]["Timestamp"]
                time_delta = time_delta.total_seconds()
            if group.iloc[ith_transaction]["IBAN"] == group.iloc[j]["IBAN"]:
                count_trx_iban += 1
            if group.iloc[ith_transaction]["CC_ASN"] == group.iloc[j]["CC_ASN"]:
                is_new_asn_cc = 0
            if group.iloc[ith_transaction]["IBAN"] == group.iloc[j]["IBAN"]:
                is_new_iban = 0
            if group.iloc[ith_transaction]["IBAN_CC"] == group.iloc[j]["IBAN_CC"]:
                is_new_iban_cc = 0
            if group.iloc[ith_transaction]["IP"] == group.iloc[j]["IP"]:
                is_new_ip = 0
    return count_trx_iban, is_new_asn_cc, is_new_iban, is_new_iban_cc, is_new_ip, time_delta


# get transactions of a given user and returns the dataset with new, aggregated, features
def create_user_aggregated_features(user_transactions):
    user_transactions = user_transactions.sort_values("Timestamp").reset_index(drop=True)
    # for each transaction of the user:
    #   - count the number of preceding transactions to the same iban
    #   - check if the asn_cc is new
    #   - check if the iban receiver is new
    #   - check if the iban_cc is new
    #   - check if the ip of the user is new

    already_seen_ibans = {}
    already_seen_iban_ccs = []
    already_seen_asn_ccs = []
    already_seen_ips = []

    for i in range(len(user_transactions)):
        # print("Creating features for transaction", i)
        if user_transactions.iloc[i]["IBAN_CC"] in already_seen_iban_ccs:
            is_new_iban_cc = 0
        else:
            is_new_iban_cc = 1
            already_seen_iban_ccs.append(user_transactions.iloc[i]["IBAN_CC"])

        if user_transactions.iloc[i]["CC_ASN"] in already_seen_asn_ccs:
            is_new_asn_cc = 0
        else:
            is_new_asn_cc = 1
            already_seen_asn_ccs.append(user_transactions.iloc[i]["CC_ASN"])

        if user_transactions.iloc[i]["IBAN"] in already_seen_ibans:
            count_trx_iban = already_seen_ibans[user_transactions.iloc[i]["IBAN"]]
            already_seen_ibans[user_transactions.iloc[i]["IBAN"]] += 1
            is_new_iban = 0
        else:
            already_seen_ibans[user_transactions.iloc[i]["IBAN"]] = 1
            is_new_iban = 1
            count_trx_iban = 0

        if user_transactions.iloc[i]["IP"] in already_seen_ips:
            is_new_ip = 0
        else:
            is_new_ip = 1
            already_seen_ips.append(user_transactions.iloc[i]["IP"])

        if i == 0:
            time_delta = user_transactions.iloc[i + 1]["Timestamp"] - user_transactions.iloc[i]["Timestamp"]
            time_delta = time_delta.total_seconds()
        else:
            time_delta = user_transactions.iloc[i]["Timestamp"] - user_transactions.iloc[i - 1]["Timestamp"]
            time_delta = time_delta.total_seconds()

        user_transactions.at[i, "count_trx_iban"] = count_trx_iban
        user_transactions.at[i, "is_new_asn_cc"] = is_new_asn_cc
        user_transactions.at[i, "is_new_iban"] = is_new_iban
        user_transactions.at[i, "is_new_iban_cc"] = is_new_iban_cc
        user_transactions.at[i, "is_new_ip"] = is_new_ip
        user_transactions.at[i, "time_delta"] = time_delta
    return user_transactions


def create_aggregated_features(dataset):
    columns = dataset.columns.to_list() + ["count_trx_iban", "is_new_asn_cc", "is_new_iban", "is_new_iban_cc",
                                           "is_new_ip", "time_delta"]
    dataset_with_aggregated_features = pd.DataFrame(columns=columns)
    dataset_by_user = dataset.groupby("UserID")
    counter = 0
    for user in dataset_by_user.groups.keys():
        print("creating aggregated features for user", counter)
        counter += 1
        group = dataset_by_user.get_group(user).sort_values(by='Timestamp', ascending=True).reset_index(drop=True)
        group_with_aggregated_features = create_user_aggregated_features(group)
        dataset_with_aggregated_features = dataset_with_aggregated_features.append(group_with_aggregated_features,
                                                                                   ignore_index=True)

    dataset_with_aggregated_features["count_trx_iban"] = pd.to_numeric(
        dataset_with_aggregated_features["count_trx_iban"], downcast='integer')
    dataset_with_aggregated_features["is_new_asn_cc"] = pd.to_numeric(
        dataset_with_aggregated_features["is_new_asn_cc"], downcast='integer')
    dataset_with_aggregated_features["is_new_iban"] = pd.to_numeric(
        dataset_with_aggregated_features["is_new_iban"], downcast='integer')
    dataset_with_aggregated_features["is_new_iban_cc"] = pd.to_numeric(
        dataset_with_aggregated_features["is_new_iban_cc"], downcast='integer')
    dataset_with_aggregated_features["is_new_ip"] = pd.to_numeric(
        dataset_with_aggregated_features["is_new_ip"], downcast='integer')

    return dataset_with_aggregated_features


if __name__ == "__main__":
    dataset = read_dataset()
    dataset = dataset.reset_index(drop=True)
    # these are users with more than 100 transactions
    users = ['003b255561d93cf8ccd1d02bc2136674', '006c6191c08277a95909a9349683297f', '00f506c5de74027518eba95f8c83c3ae', '016d54fdeda2d363ee9f26294a23d4f4', '0229eb0b5e9bc80b39137658d9ca8419', '025b22fe62a1a97c511d1f28efc31b20', '0268e1cb7f981e8c9fd858992f74e9c1', '02be1d2bb49d5da259895e1497aaaeff', '0409c7276c66cdc7595580e9c13410c6', '04d8e009145b1b5149cac8e0af4ff025', '056e96b7a3cc21aa71969a7b3d3af547', '06c11b241fbd2464d32e8d981d987405', '06cc560ea128db7d87f77648532cfa98', '07372e6249e02a5b9dd6bd00d8411869', '0744761a0d4a97d294e52ec2afc51dde', '079fee0af27f1d2a688020b0dc34d9b4', '07f4a369e34fd00d8efe9a87837766ed', '080fe25bd7874912cec352772b24dcae', '08177dac760032c95f36867355d0c720', '081897c3d2c5f72282ccc20adfe6dab5', '082ab5cb855fce641b554bdf29401326', '0867f5f05526d3b084aa69e3676f2a4d', '0919d648a05bf350eb9176ec466a8b0a', '09371191ec0fd0f0a51d9be8ee2ba3cd', '096d38441b78bcc7f7e87e3f9ba009af', '0a235e782fac179463da81690addaede', '0d0b094b1cb79bd2f32aefb580c4a84f', '0e33bcbef8719905c912e8c99d79b987', '0e55414eaf9f10b03b22692bf9edf69f', '0e65a3fa8baddb45980251d6d4856dce', '0f10d0dd83648b9c0478316d91f50f7f', '0f1a2f9e0118b7f7d391f84178b8893b', '0f4cb2d9b53fed31f36d5caee00c0875', '1060185b1b808ac0a7998b4367efd7c4', '10841d9ae906e1c828020270372344e2', '1101d241eb41f2a8f88fd0701f2b45b1', '113f0e2826eb7a4c5408e7d211825a79', '1220ed33c2d57639f8c24a7385bc4e5e', '129438a74199650fd5264bc279fd6daf', '129f2d60280c32f736653a1cc603b6a2', '12e1594858baeb941942d56145c25e0c', '12ea4d17838f9a4dc4b79489ee94c0e0', '137568440d8a2419ff4c4535c939da3a', '139e7d8af583e2bf1f51590d49948d17', '13c9f2f4167455809bf4c7fafd1e215b', '13d248c2d410193f6688b6f5f8db2042', '151e064a529aec241b229c1d21fcd79e', '169df5e434993382d054914c98ac207a', '18a9574b5380d63d590d11306c800ab6', '18c5ae99b9aef7803bec0fa8acbfdd84', '1921cac73c37d910d6cfa98a0877b5d8', '1931881718bc44830b52eb56a2d2ed8e', '19bc2ec20104cfbd30fb4bee3892c2ac', '19cd520a16f15b3a1aa5c33f144cba74', '19f6c233dc391451b32a416aa9ed323c', '1a7748b6b279e03fe8e797e939092c9f', '1a8c092938b42e392ee607a846373b62', '1aa1ae19fd9e1c41fa23e16217120702', '1aa70323fef4b62eddbb8dafec244624', '1afe68812abfe06ddf0261bc7a5b390f', '1b4e1ce2a2f74720bfe4f89094c97392', '1baea0b11d10dc85a67f5292b6d0f3a5', '1bb425c2c15b2eb345d4b5b2910621db', '1bd81073bdb0ff5fe3ea516d24594355', '1c05f0071ea0e17c32005fd9c0840049', '1c20880b33d2768c5f8f41dfc8f9dde0', '1c5ae9253b90d80deda369d683a860dd', '1caa71fe8e2445974f981f34a23cfd21', '1d6bfe77359152e1dd3c8c93a2a37a1f', '1ed695ef00fb255a1a1ba3ddde6dfc7c', '1ee9d176749b36d581bac7c8b5304a4e', '1eea381b5e736a2c15dff075b3554ac0', '1f56d13d4f9d34447c9abdaeaa96229a', '1f682685f64e4ac9634bc40206828683', '1fac5380e5795d1d1d83a2d805db3c1b', '20232c74539a771d00bdc434a47115cd', '21273a9236d9e4ee9c22d5de76e5facf', '214a4bc5fdb7d209759e8d40be0d6567', '21d971a5fa7c610aeb7efdd4e7ef618b', '220e192627b600d00eaa6b42ce8844da', '227df80b4a01e65b8974aff7694c5d09', '237331a9aa09d164864337b4a27820d5', '23ab1827b334e030ada7d1028fccaed7', '2413b32e93df0edf0ccbea839d9d0b84', '25787c1d18c5fd434ce65a646d9b446c', '2616b84eac790f81cb270088f22a98e2', '261b1482f42dd1b77a3bd679a7402e95', '26599a8b55d6f400370b14922fa7f0df', '26e3a0d922c11e2cc50d77a436743ff7', '270ad4c8593be9a04d39090be03bebad', '272a0ba5321a62c0064cef9a7eca493d', '2732d3b6ac728dc33e39829d721c1d7e', '2780b3a7abb05b9fc7bd8b4c798d43bb', '297b114fa02c4556ef7a1af60b784a84', '29cc7d6d238e70710237ea8bf6460a84', '29d9941b6efbdfcfc48464b4408b5280', '29e533751e651b60e4b3eb007318cb39', '29ea0f801fdde15ff33b35e316f7f865', '2a8121ae0c6896f89de4c33b88c83eb9', '2b06f4d3cc22a83851ee1bd5727dcc03', '2b1b09052559b1fa87d34203249f1bd0', '2c3e34a8c42822eedbdd48ce23792143', '2c84eba87ecda08c1fc6e2e492841b05', '2ca414c01128ec1c32010a36ed456492', '2ca4472117fd389c1fa414df5c7be177', '2ce967391c196ed5abbd3c76a50c3af6', '2d08990d31958f25f72cb2b29051b8ed', '2d8166134b88414ef1f0d790ea709d27', '2dc803d4ec60ee23b41a9b27bfbb7841', '2e91ceb029f4e5da13a0b481b99117a6', '2e92a74fb4ce9d2986d56b7aaf9d5046', '2f607c32eb934e72b7e6e10ef0654449', '30fb66f1f20ea19a3065142761acdfcc', '3176e1ebef38e3ee471e3a89fa2f937d', '31c1daca3f4a4238bd9de191acb57a21', '322e77775957ccbf22194fbaf94df8e0', '3291d98436049fcf353cd2f09c35657b', '32955d1348d52f5082890d688b6ee498', '3346171ba79eafa81313e39d8082703d', '33ca757cec4828a189d35492fda188c2', '33d69ffd49b9cbfb119f3615308557ab', '3521214b63803d0403c9692f6ecb2fb5', '352d735e9f56072af29d85650b32e6ce', '35f4c6565e2558be9bb31de771199867', '373fbc43f99a56c3030c4c49e3daf2b5', '37808688409649dcc216a7ec05420ac3', '379406c92e63a5b5c7b3677c3a588309', '38061b16cd78d50d1fcebba2d7980707', '39fca0a3cf04fc9e0d1ad6c67ce77f4f', '3a48f953b635a25f4f5fbb3b4e9691fb', '3ade08ddf886e01f8d92095225175f8b', '3b43aa61f32335fbfd480d19987b32ee', '3bf47cba610ca9b225dc0523a2f1d33e', '3c739651d41b06ade9c365c67f8688b5', '3c809ee390f8d8d200147cf114773110', '3cd229913dc50db0692885e1f3ed2cef', '3d01aa89ac96d0d0b73a42a68b1556e7', '3d9d15bfe2d6685f7bb26bd6612a6878', '3ec4a89981706891f74161bf2b396229', '3ec646d4d64cf173349f9b17e10b69c5', '400598252998d9dafd871aaf1b6e4ca6', '4125d52a5a4386ce7a64e038942412b9', '4146594021a4b11f5dbdba7a09e27625', '4153e0ce108b73be228cb93c2593f9e9', '4164f06f62280f4388fe59c5f8ebed79', '4197f0271eba69a5a366ce928655cbc1', '41bbba48ef9dfa34e5ba7037d9e0a773', '41da24f1c3320d4ddc1d43d66c6bf8f2', '42d6eb23855cb77b4a286e44b7afedd7', '4423323f14fdd572190c777b4c163b0b', '4502d44446b6110f99e6e6d55df0c739', '45fc7689baf3f74ce186cbfef1c04533', '4623a39bea465bf89057409189c94726', '46fa0c74a108d3341ac95718a599946c', '47f901056bd358540f85e15191ff9a29', '48638cab7beb7292b0b4c22b6d6b9b4a', '48f0d4d482e5b29dca2e120fb9f9adbc', '49399792a9bf95b9acb7b76da5276eca', '49553df019ccb70420075d06f0d13597', '497f2ddd7bfff2694f4564c6a09c7865', '49cba0a68c9e7287831f882d77ab7bff', '4a2b41af1807e2968991ccb930c8070f', '4b0dba4d27c70c19dc2451e31eacd529', '4bd99e4adceeffae72b8a755d8fbb6b9', '4bf91623182c2379a8a8ae269d4d3733', '4c2b9d0f7cba90b19c61b5da5fffefcf', '4c6ce8941ca40417981a46c63784d326', '4d26573420bc73323d99738936f65a58', '4d396d361df8b41b47f5453a9a7b947d', '4d77725d2a690541491b67c203e663fb', '4dc25be64d2fd7ace271990ec292a3a4', '4dcde28af5c9f2ecc2b50406a5c6e599', '4e62497351b1ce1184f5b087d5c643db', '4eae35a4dd3bf80e9b9cc4d67c443c64', '4fe3a65e2e80ffe20b4d4fede546248b', '5215cadd3cbe6813eb18d9fc4f86e288', '521dd402456b75b3c8a49b064e3f1c99', '52b30b8430293581a3c36af27a4d794d', '52c344475123c8d38da7d3431b024f15', '530d73a1a978a7cb10eac9abd947ac87', '53458d08e1d5ff271ca298ec1c9df847', '542aba60f688c1850f773b2f4f25f26d', '54a1a7d54370bf82933d9e9356d2a703', '54a62147e389d05e33dd7a13e659690b', '5505fc6908d31949bf7f9573bf50a295', '554393556d0c0f35c2f9f23356c453fc', '565cc90a7c23e6937f06b826c9b7ba84', '57ab44112a3342172418c6e81bfea677', '57abcdbc57d7231aa9d83da0a4b631bb', '57ff690b7b29da08764f9bd00fd1b56e', '584a0606551b0ddab140d264f588845d', '58ccef3ba1d96113dd4dd44317f15133', '58d209e58eda187bf554ffab3b03031f', '599778fdde03c4cc45a66627c6083e34', '5a3ba6fa8d765bcdf8a254cf4b8db90a', '5a45bf1914fd6f4663bdc01241ae5ee5', '5af938281e3a0bb02cc9946d369af1bf', '5b0d7b1ef0cd76d60b325482fa297ebf', '5c49097a2a7a4f013c6e46efba768deb', '5ebe3a30b486e493d1f017fbfb9fd05c', '5f04a18611e00c3db999588c8468f86e', '5f596669c91577926b54f82a563ce9df', '5fe5f89a704fd806814e4c8b9e228317', '60853b3b97f81f91dd5e0f9487e4fa70', '60b234fe5110937c1821e87e92b87a4b', '618101552d4a1e5d138e9f76cd7988cf', '619f210ab24f7f9abb54bf75e11eaea6', '619f636eeacae3a04c1bd3892b28fac5', '625492321bba7a8e7d510aadd7613e48', '62bd026a0d2862521e4366bd64743515', '62c9c37b47e8c9e91aead9f4148b0a6c', '6351ee0173b282bcd3bc9e602d8c9066', '63caee7cc5fd1b3c76dd7afd0e4434c7', '63f4b24ad39732279cbe04c8b863968d', '63faab7cb5e6e94d623809dc2d41f425', '6713f7982b60a5107b7bcef69b7be6c6', '683a349888c2faf63795bba3017db339', '683d2d9d78902f561c03a9faca5fa4f5', '68cc4e24144a41f3b9b90392262fd381', '68fa6e5bc8392cefca3d23113ccc1fc7', '69178fe97260bd4da5597fc7bc480b66', '696a60613fd58fd36c63d61add2b035d', '69e828a964eb9941fdec592831a345a5', '6a334252054233548a26e14fe1a6fd31', '6a4b476e4691a3cda0f867b14a44f77b', '6b95afde82762cb86a9258e5d5eaa74e', '6c167d8142230626b23f1a395c09b633', '6ca878219aec3bcad86c46b74dabe06a', '6d647ad6c410b485f4b6e963dd3f4676', '6d98a5f8a3df403d22a5ececc5b80655', '6e67ba79f2ed12171a67938dbe5f9b9d', '6e6ff87015b57e5c85697faf970823e9', '6e9bfe5da55b434d4371da58b86c0edd', '6e9ecf0757221f9f84e9c11fc5f9245d', '6ed72d2914eeee4af7a362c87fcf04a5', '6ef2bd86fe0de2b6d8fa732abf187888', '6fe0b4546887d8806fae5efe371f4a00', '7042d675487b2d4da1a19771d0282d64', '707bf972fd6394e5a25f6479e53d8f3a', '70b17036f76cd130ce1add37cabda23a', '70ce5ec438cc527af61284b8b0bff538', '70ee5f087be9579dd499f2432c4d67c0', '7174d9cfad9646a6561f23d5d29af631', '71be7a9ff189eddd28c5f779a887ae90', '71fdb7fcce2c2d3ec9c342373e8b2961', '72a6142b044e4a0e2955fb28141f5802', '7435df98cf80d329d0c0ca11058b244e', '74d5f108656df9f00877aee9c3ecae2e', '74e77d038f63162ee91e0920b024dc93', '752b0cd8aa9c65a628159e4b5c6c0a84', '7594d565db1dccc402cbaf956ba7cd50', '7785be915ba569363d621405889ca17e', '7793a02d7c94a12327d54a918bd5f3cc', '7838e422e74275ff2e3255a23c4fbda5', '7a2a59837293e9ee600c7c56aefaaa4e', '7a2c9db3b10b5bd6afc53a7b454e9912', '7a924d8ee97965a8d3e9b392fdc01bd5', '7b2501df1835be777d1d594bccbc8a0c', '7bc8f1e3d7938f3572e749725a1778ae', '7bfe63cf8b507aff09e59aa2327e6b4e', '7c4049869bfdcc266ea760e45edc9844', '7ce81e1e1986a323fa6286f0718e329c', '7cec13de4389d0deca5bff58f7abefe0', '7da21bd1263c90ab345391ae5e7c320c', '7de5f11d5231df0b0411bb03bf0f74d2', '7deb2be794472d0f5f1de0831ac134b2', '7e2aad413eeb6d349d978d2c00d1db3f', '7e39ad3d86bec1da43ec23813ef26e5b', '7e7bd04a5783650fbad0815a28e9b70a', '7eb1619ba92ec8f1e8f23c16804e1ac6', '7f04e225c0f9410c0daefd00ee1e5052', '7f260a20df6fe985217b7bb8a8bee9bd', '7f79d86441e4396d278a61af878ac97b', '7f891fc38ee52a55fba43bc51500ffbb', '7fa4e953241ccdc10b11f9bfa74de61e', '7fc6b2b3761a0b70e2202d1cd36f6496', '7fd19dc389f388754b79c13058ca246c', '7ff3a7f0e801509b3d8e18682ca76e77', '8015cc7b96e82ff7da2dcce4e2d291c8', '80284c73fa77108b49aace97c1c50576', '80293ff0392587375fac62c75ae461da', '804b134bbf40670c40fcca4607a0af3d', '8129700be2f5746fd5663e165e22ad22', '813f8083d91e7ad7420762a8b48278a1', '818a285e006953c2bf5c1cb0fa94012e', '8225ea4ee44f2215f629f61a95f9d3c6', '8237aa40217649ed70f9ecf828b1ceca', '829d2d8abe622ef19cde178596ae7af0', '833a2a96eb5fa8a9fd98f44d2c167e47', '83530c6140d4806d87a13c9da5f45110', '83e464735d321ad83eb1a2d242e67e00', '83ef056c9db309774396e36abbb86f0d', '84860460b6eaab988fd4c546803437a7', '8503b475a69df20dad5ea888b84bed7a', '85b0f0f37a21d4a64ecde47cfe42cc06', '8624b906c404e4c312ae7a23fb689c34', '866263812752752f98e258f82e17b2ae', '872a8ad6b3aa5416eb1d0be74d001bf5', '87bc5832857c27ddc07fc9020f2bc347', '87ff9aacfcbe8bb87aa9a453426b93b9', '882f6fc8b756e09cf81fa932cf6a7e6b', '887e8a839f5f137029a46c415621e45e', '8894694e2ed0bbbb90e25a87c308579a', '88e2f4d1ef5513aba0694aeb7c6e3a91', '897d673e7982b1eb5eafb193031c4d74', '8a1a8a484c4c80214ccf94151c164faa', '8a296a30d34524b1ae69111a9aaf4c57', '8a51ad86b3e3b6ff670cb973bcbea648', '8aa8d45e285d2115b1e05a0110a23158', '8bebbe5d98fc6d86a79c4e7d161b9834', '8cadaa9347d3bfd4a91e5d8c9203d09a', '8cd3c08cf16228398764174755bf3260', '8d58800af540875853bed80863c5b9f7', '8d8065fcfe392b69a564f9b7f8ac35a0', '8e75ccf4c128d15667edde3e89d7f3f7', '8ec99c1f30876c1c6315e989c6e30f39', '8eed7749d349a702bda0dd7fa266bfe5', '8f3a7033886f385aa817a05033ef8a92', '8f3c302907e294a5085347f39919a0ac', '8f801c3cee6edbc87663665d25a8d274', '8fa91c1b0b6de5e6a082ef9f388f36b3', '8fb515a64dedb8ddfc04d3d610bf0b6f', '8fd771b58e732aff031c737d701439d4', '8fef2b690aa2e4b390768bde27a7b3c3', '900b7d6c2f9666d7375bee0b67250a95', '91108cc5bd038b363efe5367bf606922', '9174ef4b45a99feba0c893c96bf80e3e', '93f20db7405dcb4a57aa32df0506276a', '94d9116e053c7ee8e11de282caf25aa0', '95a30f59e465ab00d8828735163c8b40', '95e837acebd3683dcf70517574b8f025', '97c07071469dfe607373186a61c37b3d', '97d1c77cb77d202a7f1a5aab13a8361d', '97d8acecea4c52057760c684c508dd73', '987b0e482c2d32e7bca5d9b4b622b3b8', '98c4d98c6f0663a7a3b4e337b66698e7', '994241062daa5a37d6229dba691f3703', '99ba0de95080b39fd4030be1bab22372', '9ac9cce215447a4a89a2d4280f283d74', '9af519cc161b3ab4b6c95ffdf11c9ea7', '9b05072fe2e1d9a4a9ee14e5c04f071d', '9b0784cf039acbd966b30583772dc4ce', '9b09d1193fa4d6d53249572fee197ac9', '9b0e3eb39e5cadcb476b3fe9a2a9fed5', '9b6d55ea75451e53aa86ac7b7395b1c0', '9cc50798e32205e427b3a82c35737478', '9dd86a84ccbc5ca6c0067a02a1e53e00', '9eea7ad7ddf672582aef873839c8d10b', '9f166bceeaf04e95dad81bcca3cfa29c', '9f509f80ff65f2f09904d548e89dea59', '9f87d379773076b3cfde4f67da8d2634', 'a136b022d4cd6c95e696ac498b09d50e', 'a15e2879a9d47caff00ac81c84614cd6', 'a1edbfdc96018463b6e4991938925df9', 'a205e0ca260b713013d6212509a7581f', 'a2756a4678fff7a48e63a5921aff55c7', 'a29341a1e26a7170fb6d18fea6e5e1f0', 'a30b92774149292dd426cb208e1af147', 'a32d501eeae6a2ca0dc89d5396787bfd', 'a389b22c08bad6b4a19213230cedfe7b', 'a38c5268e0f8dd13acf95eabeb82a83e', 'a39aa4c087e193b33c71afd9792a5317', 'a3e4911b8de4a8f0e8152ace48830c09', 'a448328f5bca4d70c827046ca0ce139f', 'a53fd4372e829f8713c9cb2e3a956214', 'a674db0a8fbb47acdbbb87d461fd8966', 'a6908ab71d3a661edd7f5e626f73b734', 'a694474ffc97347ce3e850747e33f0cf', 'a71c4bd6370092dbffd9dcc9fe4765e9', 'a7a5235566dc92c817bcf289660e32a4', 'a7eecc077ef9ffb9df8cc2c3075bb2e4', 'a81597d4c1a103639bc020563d8f4d76', 'a83132078295d6bb99c7598ddea2fbba', 'a861d1375631042921eb9ef84ef91b6f', 'a86f49cfbffdba2d02e6ffc5427b1f0f', 'a89ee46d0fff7adda18df557ea9a7f37', 'a91b56d66d477caa22a2d30ee8d6d86f', 'a997cd16ef0fad48f605818946b356cf', 'aa73bc4d6e27f46e2788b1c2af96b1fb', 'aad1057ff459268156f141a09dfa756d', 'ab529cdfc0cf9e06bab8d218d36523de', 'ac24174a52d264bc957be5d597773bd2', 'ac404fc573cb4cc87be4fda3107f057d', 'aca754a700be29d91c37cb55ee06b7f0', 'ad22ba2f5d81cd5599e54f7e83a6e872', 'adb6bd938e0e6a2fc2aa0279f0ce9494', 'addc7fe0f8e2606fb358d854692c56e4', 'ade7ed6480fc0a27edc08ab64f4ec4c7', 'ae70be257a2ac8a251faa8b8048722e7', 'aeabd08fc9a49a4e445baee1dbfae4aa', 'aed832895d17944bcc0a66f0d255ad51', 'aef39eb5c8439046a23099bfda36b73a', 'af0991eaaacf49f84106c3d204ad430e', 'af2515b084d8c261d2499e0f376219d4', 'af4b34206a0ecb0c324a684d46b3ceed', 'af55dca6b9a2e4430370bf4e260a0563', 'af9f1fd2002c3e426e5a2fcf0c10d80d', 'afcbf06fbe767ea95a6f3e1ce1c4c4c7', 'afe049f3e30fc64bfe163e8f3e67fe42', 'b05dad86a81b971169736ebd1bc9f682', 'b0997917bd6e0135e432634933516090', 'b09cc86335256bad0a337fd087836122', 'b0e997b06b8f27e35b74b9ee3d4379e4', 'b17c6448900ac4407b6653eb93a9c26b', 'b1f258f4654237006b4823a68af53d4e', 'b2630314771ab407b0997621022bb9ad', 'b281aa7f775de0200cb9bd830f1ead7a', 'b2bf76bada84fc5c86f68c24d2c7aa2c', 'b3864ef210310cc184d11330d846990f', 'b448fc61d9ff4c3f7aa810dfc3e7898e', 'b4c861f3835c7c3a6f2c5b93d2bf54df', 'b4ef80e85930be2dd0aab1bfaef352ae', 'b4f309cf6bb5bb4a78f613f8b5eab61f', 'b4fe4bc82e29bb2712cc63a2f183b0b5', 'b5b4abc633272ead1880ed6ba351db6e', 'b64fcc0305a450244356b1e9d2f7c89b', 'b71cff489ae4d658966815b9b039be74', 'b86db18d1d5abc418b4d33f0d585d7fe', 'b892b720c690dadc11f3faf1213b8bf3', 'b8b022cd25202671537bedfe7c2c69c3', 'b94db4071b61a2ec509d950e347c6eca', 'ba1d188757350440de0a630573158407', 'baa2a4999da8d7740f09d6b98732ba0f', 'bacc75082edf992fbecf549610604e4a', 'baccdb078c5f0940843654fc060be6c7', 'bc716423a427f3c725f73fa7456f0d52', 'bd83c9ab19954f8d65064ee9aa7b2034', 'bda826052312ce297abf2f1f51b3c040', 'be0fd01382d30d2ad32abea5dae246c7', 'bebadbd32dde002bef17b3b385f87fab', 'bf3eee7fbbdddd65e91f4b62d23ea349', 'bff82e3291c16f9299656e1c9cf0a5e6', 'c0511c0d7e42b3a979599a082d9c04ea', 'c06da86f8c3a41f5fc78da302a4ab7be', 'c11ec14e26cbfa0e28413fe1f68feb23', 'c2482c4d96700150796a0082830b0b26', 'c26bc24551adab52cdcdafea9cb47090', 'c2b335018fb89785b25d1997b6ff2752', 'c48ba44bccc990da33392d3591e01968', 'c49e28a2a20fb8b708742ad29392154a', 'c4ee9d459a2f02dc1eb61978857a3c4e', 'c5d870f255f90fc4393b7b9396a5464f', 'c730cf335cbe9181776a1b542917c080', 'c75ce237acb5e7d2e74e8329713e09b9', 'c779dfba2a7ed95c3344eaa53d64eb53', 'c79e23c4bbccd247f86a18125a7893f1', 'c80df9f6fd9e08b75a097a458af7c535', 'c886672227282bf5ed9d55112e0e54e7', 'c8df09be008e3bd26ed03dc787c28461', 'c92ac292e782ce3571f1dc1819876f82', 'c9501c9829d2c5fab5e9bfb8e8ec91aa', 'c96438f3b7c39e9561939a638ee3e631', 'c9c781cb1207e6d2eb0af98e664349a3', 'c9faa81f85f47ee3e618a80ad6af0c45', 'ca6ce6297dea13f6dbf017217eae9be5', 'ca7f9e9bfbccccad7de5931a2995fa03', 'ca9a6f8e6a2249f267fe4895e1e4a8f4', 'cab6334e87472911f08b2798124e8d4a', 'caef7f1134bfea1e5259e962eb817c0a', 'cd605d6eae446433063d7c6aaf99835e', 'cdcece82902b25493c364a69ea08a5ed', 'cdecff10de79bfd970b53b4e51a7c28c', 'ce0bc76ed60deccd73e0d1d704521dc7', 'ce14e89a137cd1ccd4145a316584a906', 'ce55fea4974a8047d73a4c5fe9d1050d', 'ce98a7df704b77637f1a5e3746fb4c70', 'cf2e0d8bfb6bb77d959113e47b734585', 'cfc92809c6ad7f1d64efe6870b9ba469', 'd0395f0b6a90e48bff4f12fbe309038c', 'd0d28adab71a57a142ab69e094aa9ef2', 'd10f00c2a0e426cfcf8f6b818ee64907', 'd15d244a943443e436bb0f44ef5abba7', 'd237049bb8873363845788ce94d20078', 'd2a99c6d753f4db73c1a520699185216', 'd382bbe75671244c108100269b58dd22', 'd45bcdea1a7747dcb045164be6fd71eb', 'd47928417bd21ab8df82ffd86b954149', 'd4df9a0f3f99cd69d5f1207130232def', 'd52b6f04de58422a59df3b02127eb249', 'd5d081148ea1f838174a8644c691e79f', 'd5ede3239fa9590e3c2319f3fa29444f', 'd6d9e67ddc4753e819b4fe52136bdc1e', 'd70aacf37e6c3b78ad82cad3c3483a3e', 'd70f57b9e07f2aa34e2b2c91860c5e03', 'd73bd7d09ea9c27b6bfea17aabd33e26', 'd786df842b00ae3a2566396a88d6804d', 'd79d855229df5bce728330680bf3194b', 'd7ae44433bb16cf932d2911546d44c13', 'd88aa9fa459bfe9c8825798c16d0c5f8', 'd88c80787eb89e1cc347f2b204578f69', 'd97924c40ac5219e5f2bb555f90c6e2c', 'd9f6ea1b192bcf7640097118962207b4', 'da7c1cb535b5d1270cbdb1da46a584b0', 'dabce4b3ae67b6bc4ecbbed76f5bc065', 'dbb8c250c1df43caac564413c8e14fcf', 'dd457572476495ec60828bbca41f2ce3', 'dd5b4ef0b1570819e9fc52ac5ec29631', 'ddeb93f745799756321111c36da5413a', 'de200d94b65bbd0a6665cff2f427fb1a', 'de7b1a6adc0e8133d324c9bb91898d6c', 'de99c74be6be0fa1e7c0bf9cd3803a54', 'deb2e94c747b446de80178ddeba909f9', 'dee99bb3f905c13dd6d25212d5f88ad5', 'df796e05d87b4ad21ef991387687e4c2', 'e0118e631c402912ff3665b78cd42477', 'e02b77b5fd19f007665f6c4b03c9a498', 'e0a7f18ade248ca00e98c488113f3038', 'e1a752b712fced218563716846dc6d36', 'e209ea9f9188f16b849731c16f0b0dcf', 'e219bf569bde67de24f8ea78731234b3', 'e2443f6234a35bc6111836cdbbaa2374', 'e27eeb1c6634160a7250182bad6a174a', 'e402a8be28f4c08f38f2a5e3f819c7b6', 'e46d7d8f16507225bcb3091b2b43b2e9', 'e4cf93af3848549f91d30a9cb96438d5', 'e4fbf62987648e19f7b6cf02c6210bdb', 'e5cf69895acb9cbda07273abd35eac54', 'e603600f7c7290d0d88dd559fcd9e796', 'e6b0ac1a898933bef3571da46a27aacd', 'e778b5b5086d58e330cad42a65997993', 'e7bfcf859f931dff46adceee012f5ede', 'e7e2e4d7af23dfda6a3d3487b4fb4d07', 'e82954f66917a8c5dadc00680b8a5f01', 'e8fa1826555ad408092b0b803c1b5391', 'ea339967dd820d013e4f03d87bb9e8e1', 'ea51e010e19e08cc0a59d097aa24b059', 'ec96d979b693f30057feb958e5519978', 'ecd6d604efd5360018d1fd9cdbf855b5', 'ece322747db16adec283252898d8854e', 'ed101997588746185f77d48b3da4403b', 'ed1b47b17000e256e5fc36bd311baecc', 'ed6e15388ea2ef408728f00a22147699', 'eded5ca42ae667129d58b6a7bc3cd550', 'eee74e177e8702ba667d8a6998c36b24', 'eef554a8159936fe0a6c957320996554', 'ef2c6ad17139d71b36313cfb1ff4db0c', 'efa38329a7b93ce3c17b15fddbcc49f5', 'efb8b21230452523b6e033c5ec9f6886', 'efc2178e75d6b9ce69baac2c35d0f10a', 'f060d384a8ef3ef6b401d38314a22f3c', 'f0c2d1f88b49e261eaeb2a830bb8898a', 'f0cdd92a4283b6d6f2294ebd4f780d66', 'f10631610b477a1e247cee977d1f6dc0', 'f13ac3ad54e314aeea70184d052eeb30', 'f14811ddc4ee7ebb434de7f15d846d59', 'f1bbac4bea66cfbf0c252c4717ecf3e1', 'f1d149c81e3d3af14bbed2df6fc33152', 'f22ceccbda50092f47effa057ac3b3c8', 'f234cbe41becb2a92c5d0492780688b6', 'f27185b792694c90d362cb1200a4f70e', 'f2a7341750c1cc6dc8bea45185a7fe26', 'f3072f0c3c170abd5a683c949e4d6ce5', 'f34d4b9354707902893a6866cb489ab6', 'f36055c3798f74fb008fe47b8fe4af20', 'f368923c8525b40b9fa9138d10d1affb', 'f37bc2e30271d18fbfa2a265e9c681a2', 'f3a752916cf26f1c138c4e0ea40dfe3e', 'f3ab19fae9ee8fa136fdcbab602bc215', 'f424eca19ac27edf54cf27d8b45437b9', 'f424ef46ea106882fc1a73e9fde1c15b', 'f4584918bc8e4e5415c15486f7b71e3c', 'f45dfa527bcf94008e6829d6a10a93b0', 'f481fab56f5ddf2f126c0baea7eaab9c', 'f4c33aa6943784d70113b9a42864c1c8', 'f604359861f4e5120e408624c4a8303b', 'f61328987c6693767808be9594ba4732', 'f62708349a2e7201440b96dee43982f0', 'f6595c35949faeef95cee03a8bb188c5', 'f7f9b6758f79f44ec372023e8ef7fde5', 'f83e191463e727cc5cbfca453ac7eb96', 'f8bccaab184c8f83bbd7102b46326f8c', 'f94d9584f2e3a119d5118f85a5386ed5', 'f98b7e55c879e3c0b0170bbc64a38ee3', 'fa241af7ef47d252e6bbeb2138e1bc32', 'fa4ccede555aa229413b5c5037283821', 'faf236abafd765cbbd97b20609ec01ff', 'fb450b1ee21a923f56d08bcbfff6b2d8', 'fb5425215bbabb6e1dca5dd951e4eca5', 'fbf3d5c20708c03da0a63d69dab9d923', 'fc1490f3ef44fd80cd7045e198a9e0c7', 'fca8c64b93a5a55fadc6c8a9f0358db7', 'fd26e52dc361d9b34ec74e296a0d8fb7', 'fdbe4b7e23d4e9b23a29441564c00d6a', 'fde3fbdebb72bdceacdb0f950222c300', 'fe4cffe4ca896c5d2faabac00238086c', 'ff65f4b560adee317c7eb0f5bf7d8d1b', 'fffddb7c733c73bfc76c2822a17e84e4']
    # users = random.sample(users, k=100)
    print("In total, there are ", len(users), "users")
    # considering only users with more than 100 transactions
    dataset = dataset[dataset.UserID.isin(users)]

    thirty_days = timedelta(30)
    first_date = min(dataset.Timestamp)
    last_date = max(dataset.Timestamp)
    last_date_train_set = last_date - thirty_days

    dataset_by_user = dataset.groupby("UserID")
    # get only users that has more than 5 transactions in test set
    dataset = dataset_by_user.filter(lambda group: len(group[group.Timestamp >= last_date_train_set]) > 5)

    print("After removing users with less than 5 transactions in test set, there are", len(dataset.UserID.value_counts()), "users.")

    dataset_train = dataset[dataset.Timestamp < last_date_train_set]
    dataset_test = dataset[dataset.Timestamp >= last_date_train_set]
    print("Injecting frauds in train set...")
    dataset_train = inject_frauds.craft_frauds(dataset_train, dataset)

    print("Injecting frauds in test set...")
    dataset_test = inject_frauds.craft_frauds(dataset_test, dataset)

    dataset = dataset_train.append(dataset_test, ignore_index=True)
    dataset = to_boolean(dataset)
    dataset = create_engineered_features(dataset)

    dataset_train = dataset[dataset.Timestamp < last_date_train_set]
    dataset_test = dataset[dataset.Timestamp >= last_date_train_set]

    # NB: Creating aggregated features because at runtime the model has to decide if the transaction is a fraud
    # and in that case, the fraud must not be considered in the history of the user
    print("Creating aggregated features for training set...")
    dataset_train_with_aggregated_features = create_aggregated_features(dataset_train)

    print("Creating aggregated features for test set...")
    dataset_test_with_aggregated_features = create_aggregated_features(dataset_test)

    # saving the dataset
    dataset_train_with_aggregated_features.to_csv(
        "/home/mpapale/thesis/datasets/bonifici_engineered_train_" + str(len(users)) + "_users_all_scenarios.csv",
        index=False)
    dataset_test_with_aggregated_features.to_csv(
        "/home/mpapale/thesis/datasets/bonifici_engineered_test_" + str(len(users)) + "_users_all_scenarios.csv",
        index=False)
