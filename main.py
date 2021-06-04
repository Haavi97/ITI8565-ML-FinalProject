from multiprocessing import Process
from pandas import DataFrame

from data_loading import load_data
from plot_initial_data import histogram, histogram_stack
from data_analysis import correlation_analysis

if __name__ == '__main__':
    df = load_data()
    print(df)
    Process(target=histogram_stack, args=(df,)).start()
    Process(target=correlation_analysis, args=(df,)).start()
