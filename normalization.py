import copy
import numpy as np
from sklearn.preprocessing import normalize

def nan(df):
    total_nan = df.isnull().sum().sum()
    percent = int(100*total_nan/df.shape[0])
    print('\n********************')
    print('There are {} NaN values in the data frame ({}%)'.format(total_nan, percent))
    print('NaN value per feature:')
    for column in df:
        col_nan = df[column].isnull().sum()
        print('\t{:<20}:\t{}'.format(column, col_nan))
    data = copy.deepcopy(df)
    data = data.fillna(df.mean())
    return data

def no_nan(df):
    return df.dropna()#subset = df.columns, axis=0)

def normalize_unit(df):
    return normalize(nan(df), axis=0)

def normalize_no_nan(df):
    return normalize(no_nan(df))

def normalize_separately(df):
    result = np.empty(df.shape)
    print(result)
    i = 0
    for column in df.columns:
        current = normalize(no_nan(df[column].reshape(1, -1)))
        print(current)
        result[i,:] = current
        i += 1
    return result


