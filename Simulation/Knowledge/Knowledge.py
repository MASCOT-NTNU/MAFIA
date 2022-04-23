"""
This script only contains the knowledge node
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-21
"""


class Knowledge:

    def __init__(self, gmrf_grid=None, mu=None, SigmaDiag=None):

        self.gmrf_grid = gmrf_grid
        self.mu = mu
        self.SigmaDiag = SigmaDiag


