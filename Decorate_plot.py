import seaborn as sns
import matplotlib.pyplot as plt

def decorate_plot(ax):
    sns.set_theme(style="ticks")
    ax.minorticks_on()
    ax.tick_params(direction="in", right=True, top=True)
    ax.tick_params(labelsize=18)
    ax.tick_params(labelbottom=True, labeltop=False, labelright=False)
    ax.tick_params(direction="in", which="minor", length=5, bottom=True)
    ax.tick_params(direction="in", which="major", length=10, bottom=True)
    ax.grid()
    return ax
