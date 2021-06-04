from multiprocessing import Process
from pandas import DataFrame

from data_loading import load_data
from plot_initial_data import histogram, histogram_stack

if __name__ == '__main__':
    df = load_data()
    print(df)
    histogram_stack(df)
    # for column in df:
    #     Process(target=histogram, args=(
    #         df[column].to_numpy(), column,)).start()
