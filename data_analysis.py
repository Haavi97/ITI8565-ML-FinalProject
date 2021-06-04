import seaborn as sn
from matplotlib import pyplot as plt

from plotting import maximize_screen


def correlation_analysis(data):
    corr = data.corr()
    print(corr)
    ax = sn.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap=sn.diverging_palette(20, 220, n=200),
        square=True,
        annot=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    maximize_screen()
    plt.show()
