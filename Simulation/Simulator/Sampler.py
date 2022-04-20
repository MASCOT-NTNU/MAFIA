"""
This script samples the field and returns the updated field
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-18
"""


from usr_func import *
from sklearn.metrics import mean_squared_error
from MAFIA.Simulation.Kernel.Kernel import Kernel


class Sampler:

    def __init__(self, knowledge, ground_truth, ind_sample):
        self.knowledge = knowledge
        self.kernel = Kernel(self.knowledge)
        self.ground_truth = ground_truth
        self.ind_sample = ind_sample
        self.sample()

    def sample(self):
        self.knowledge.trajectory.append(self.knowledge.current_location)
        # eibv = self.kernel.get_eibv_1d(self.ind_sample)
        # dist = self.getDistanceTravelled()
        # print(self.ground_truth.)
        self.knowledge.spde_model.update(rel = self.ground_truth[self.ind_sample].reshape(-1, 1), ks = self.ind_sample)
        self.knowledge.mu_cond = self.knowledge.spde_model.mu
        self.knowledge.Sigma_cond_diag = self.knowledge.spde_model.mvar()
        self.knowledge.excursion_prob = get_excursion_prob_1d(self.knowledge.mu_cond,
                                                              np.diagflat(self.knowledge.Sigma_cond_diag),
                                                              self.knowledge.threshold)

        self.knowledge.previous_location = self.knowledge.current_location
        self.knowledge.current_location = self.knowledge.next_location
        # self.knowledge.rootMeanSquaredError.append(mean_squared_error(self.ground_truth, self.knowledge.mu_cond,
        #                                                               squared=False))
        # self.knowledge.expectedVariance.append(np.sum(np.diag(self.knowledge.kernel.Sigma_cond)))
        # self.knowledge.integratedBernoulliVariance.append(eibv)
        # self.knowledge.distance_travelled.append(dist + self.knowledge.distance_travelled[-1])

    # def getDistanceTravelled(self):
    #
    #     return dist


