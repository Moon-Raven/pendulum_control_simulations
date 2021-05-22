import matplotlib.font_manager as fm
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.integrate
import pendulum

DEFAULT_WIDTH = 12.12364
DEFAULT_HEIGHT = 6

def set_general_figure_parameters():
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Computer Modern Roman']
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['axes.labelsize'] = 8
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    mpl.rcParams['legend.fontsize'] = 7

def new_figure(width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
               tight=False, constrained_layout=True, subplot_count=1):
    set_general_figure_parameters()
    INCH2CENT = 2.54
    figsize = (width / INCH2CENT, height / INCH2CENT)
    fig, ax = plt.subplots(subplot_count, figsize=figsize,
                           constrained_layout=constrained_layout)
    if tight:
        fig.tight_layout(pad=0)
    return fig, ax