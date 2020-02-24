from imblearn.over_sampling import SMOTE
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from dataset_creation.constants import *

def oversample_set(x, y, ratio=0.1):
    print("RESAMPLING DATASET USING SMOTE")
    shape = x.shape
    x = np.reshape(x, (shape[0], shape[1] * shape[2]))

    if DATASET_TYPE == REAL_DATASET:
        print("Size classes before applying RandomUndersample: ", np.unique(y, return_counts=True))
        rus = RandomUnderSampler(sampling_strategy=0.005, random_state=42)
        x, y = rus.fit_resample(x, y)

    over = SMOTE(sampling_strategy=ratio, random_state=42)
    print("Size classes before applying SMOTE: ", np.unique(y, return_counts=True))
    x, y = over.fit_resample(x, y)
    x = np.reshape(x, (y.shape[0], shape[1], shape[2]))
    print("Size classes after applying SMOTE: ", np.unique(y, return_counts=True))
    return x, y