""" Put the definition of a cell in its own file for imports reasons.

"""


class Cell():
    """ Class representing a cell in the inversion grid.
    A cell is just a cuboid, so all we need are the min and max along each
    coordinate. Distances should be in meters.

    Parameters
    ----------
    xl : float
        Minimum along x-axis (l for low).
    xh : float
        Maximum along x-axis (h for high).
    yl : float
        Minimum along y-axis (l for low).
    yh : float
        Maximum along y-axis (h for high).
    zl : float
        Minimum along z-axis (l for low).
    zh : float
        Maximum along z-axis (h for high).
    """
    def __init__(self, xl, xh, yl, yh, zl, zh):
        self.xl = xl
        self.yl = yl
        self.zl = zl

        self.xh = xh
        self.yh = yh
        self.zh = zh

    # Redefine printing method so it display useful informations.
    def __str__(self):
        return(
            "xl: {}, xh: {}, yl: {}, yh: {}, zl: {}, zh: j{}"
                .format(self.xl, self.xh,
                        self.yl, self.yh,
                        self.zl, self.zh))
