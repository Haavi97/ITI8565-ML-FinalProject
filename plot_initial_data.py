from matplotlib import pyplot as plt
from time import sleep


def histogram(column, tstr):
    plt.hist(column)
    plt.title(tstr)
    plt.show()


def histogram_stack(df):
    ncolumns = df.shape[1]
    half = ncolumns//2
    fig, axs = plt.subplots(2, half)
    i = 0
    j = 0
    for column in df:
        k, l = i // half, j - (j // half)*half
        axs[k, l].hist(df[column].to_numpy())
        axs[k, l].set_title(column)
        i += 1
        j += 1
    fig.suptitle('Column histograms')
    fig.show()
    sleep(30)
