"""
This script includes all essential functions for the kernel funcation
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-21
"""

from scipy.stats import mvn, norm
import numpy as np


class Kernel:

    def __init__(self, knowledge):
        self.knowledge = knowledge
        pass

    def get_eibv_1d(self, ind_k):
        Variance = self.knowledge.spde_model.candidate(ks=ind_k) # update the field
        EIBV = 0
        for i in range(self.knowledge.mu_cond.shape[0]):
            EIBV += (mvn.mvnun(-np.inf, self.knowledge.threshold, self.knowledge.mu_cond[i], Variance[i])[0] -
                     mvn.mvnun(-np.inf, self.knowledge.threshold, self.knowledge.mu_cond[i], Variance[i])[0] ** 2)
        return EIBV

    def get_excursion_prob_1d(self):
        Variance = self.knowledge.spde_model.mvar()
        excursion_prob = np.zeros_like(self.knowledge.mu_cond)
        for i in range(excursion_prob.shape[0]):
            excursion_prob[i] = norm.cdf(self.knowledge.threshold, self.knowledge.mu_cond[i], Variance[i, i])
        return excursion_prob

    def get_excursion_set(self):
        excursion_set = np.zeros_like(self.knowledge.mu_cond)
        excursion_set[self.knowledge.mu_cond < self.knowledge.threshold] = True
        return excursion_set



