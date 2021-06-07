from multiprocessing import Process
from pandas import DataFrame
from sys import argv
from matplotlib import pyplot as plt
import numpy as np

from data_loading import load_data
from plot_initial_data import histogram, histogram_stack
from data_analysis import correlation_analysis
from feature_selection import pca_analysis, selec_best, print_best, select_and_print_best
from normalization import nan, no_nan, normalize_unit, normalize_no_nan
from clustering import clustering
from decision_tree import decision_tree, tree_depths
from nnclassifier import nn_do

if __name__ == '__main__':
    df = load_data()
    input_ = input('Normalize:\n' +
                   '\t1. With Na\n' +
                   '\t2. Without Na')
    data = normalize_unit(df) if input_ == '1' else normalize_no_nan(df)

    df2 = no_nan(df)
    
    nfeatures = df.shape[1]
    tags = list(df.columns)
    
    labels = np.array(df['Potability']) if input_ == '1' else np.array(df2['Potability'])
    # labels = np.array(data[:,-1])
    X = data[:, 0:(nfeatures-1)]
    if len(argv) == 1:
        print(df.info)
        Process(target=histogram_stack, args=(df,)).start()
        Process(target=correlation_analysis, args=(df,)).start()
        Process(target=pca_analysis, args=(data, nfeatures,)).start()
        Process(target=clustering, args=(data,)).start()
    elif argv[1] == '-presentation':
        if input('Next?:') == '1':
            print(df.info)
        if input('Next? histogram:') == '1':
            Process(target=histogram_stack, args=(df,)).start()
        if input('Next? correlation:') == '1':
            Process(target=correlation_analysis, args=(df,)).start()
        if input('Next? PCA:') == '1':
            result = pca_analysis(data, nfeatures)
            print('Getting results from PCA:')
            print(result)
        if input('Next? Clustering:') == '1':
            Process(target=clustering, args=(data,)).start()
        if input('Next? Tree:') == '1':
            Process(target=tree_depths, args=(X, labels,)).start()
            depth = input('Enter tree depth')
            decision_tree(X, labels, depth=int(depth))
        if input('Next? nn:') == '1':
            epo = input('How many epochs?')
            accs, loss_ = nn_do(X, labels, nfeatures-1,
                                epochs=int(epo), lr=0.01)
            plt.plot(accs, label='Accuracy')
            plt.plot(loss_, label='Loss')
            plt.title('Performance per epoch')
            plt.xlabel('Epochs')
            plt.legend()
            plt.show()
    else:
        select_and_print_best(X, labels, tags)
