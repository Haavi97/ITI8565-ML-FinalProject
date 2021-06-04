from multiprocessing import Process
from pandas import DataFrame

from data_loading import load_data
from plot_initial_data import histogram, histogram_stack
from data_analysis import correlation_analysis
from feature_selection import pca_analysis
from normalization import nan, normalize_unit
from clustering import clustering

if __name__ == '__main__':
    df = load_data()
    nfeatures = df.shape[1]
    print(df.info)
    Process(target=histogram_stack, args=(df,)).start()
    Process(target=correlation_analysis, args=(df,)).start()
    data = normalize_unit(df)
    Process(target=pca_analysis, args=(data,nfeatures,)).start()
    Process(target=clustering, args=(data,)).start()
