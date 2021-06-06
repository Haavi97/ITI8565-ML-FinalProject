from matplotlib import pyplot as plt

def maximize_screen():
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()