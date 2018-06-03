"""Tools for testing and debugging pyqtgraph plotting of data"""

import pyqtgraph as pg


def popplot(x, y, invertx=True):
    """
    Pop up a pyqtgraph plot of x, y.
    :param x: numpy array of x coordinates
    :param y: numpy array of matching y coordinates
    :param invertx: Boolean indicating if x axis shoudl be "flipped" NMR-style
    :return: Pop-up graph of x, y data.
    """
    popup = pg.plot()
    popup.getViewBox().invertX(invertx)  # Reverse x axis "NMR style"
    popup.plot(x, y)
    # noinspection PyUnresolvedReferences
    pg.QtGui.QApplication.exec_()
