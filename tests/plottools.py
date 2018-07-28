"""Plot utility for human-evaluated tests of data plots."""

import matplotlib.pyplot as plt


def popplot(x, y, invertx=True):
    """Create a pop-up plot of x, y data.

    Arguments
    ---------
    x, y: np.array-like objects of x and y coordinate data.
    invertx: invert the x axis if true (the standard for NMR spectra).
    """
    plt.plot(x, y)
    if invertx:
        plt.gca().invert_xaxis()
    plt.show()
