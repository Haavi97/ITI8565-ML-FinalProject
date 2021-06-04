import copy

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