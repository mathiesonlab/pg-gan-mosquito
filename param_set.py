"""
Utility functions and classes (including default parameters).
Author: Sara Mathieson, Rebecca Riley, Jacky Siu Pui Chung
Date: 5/27/25
"""

# python imports
import numpy as np
from scipy.stats import norm
import sys

# our imports
import simulation

class Parameter:
    """
    Holds information about evolutionary parameters to infer.
    Note: the value arg is NOT the starting value, just used as a default if
    that parameter is not inferred, or the truth when training data is simulated
    """

    def __init__(self, value, min, max, name):
        self.value = value
        self.min = min
        self.max = max
        self.name = name
        #has effect on the speed of discriminator become confused and reach stable competing phase
        self.proposal_width = (self.max - self.min)/15 # heuristic

    def __str__(self):
        s = '\t'.join(["NAME", "VALUE", "MIN", "MAX"]) + '\n'
        s += '\t'.join([str(self.name), str(self.value), str(self.min),
            str(self.max)])
        return s

    def start(self):
        # random initialization
        return np.random.uniform(self.min, self.max)

    def start_range(self):
        start_min = np.random.uniform(self.min, self.max)
        start_max = np.random.uniform(self.min, self.max)
        if start_min <= start_max:
            return [start_min, start_max]
        return self.start_range()

    def fit_to_range(self, value):
        value = min(value, self.max)
        return max(value, self.min)

    def proposal(self, curr_value, multiplier):
        if multiplier <= 0: # last iter
            return curr_value

        # normal around current value (make sure we don't go outside bounds)
        new_value = norm(curr_value, self.proposal_width*multiplier).rvs()
        new_value = self.fit_to_range(new_value)
        # if the parameter hits the min or max it tends to get stuck
        if new_value == curr_value or new_value == self.min or new_value == \
            self.max:
            return self.proposal(curr_value, multiplier) # recurse
        else:
            return new_value

    def proposal_range(self, curr_lst, multiplier):
        new_min = self.fit_to_range(norm(curr_lst[0], self.proposal_width *
            multiplier).rvs())
        new_max = self.fit_to_range(norm(curr_lst[1], self.proposal_width *
            multiplier).rvs())
        if new_min <= new_max:
            return [new_min, new_max]
        return self.proposal_range(curr_lst, multiplier) # try again

class ParamSet:

    def __init__(self, simulator):
        """Takes in a simulator to determine which params are needed"""

        # const (mosquito right now)
        if simulator == simulation.const:
            # using 500k - 3M based on estimate of roughly 1M in 2017 paper
            self.Ne = Parameter(1e6, 500e3, 3e6, "Ne")
            self.reco = Parameter(8.4e-09, 1e-9, 1e-8, "reco") # stdpopsim
            # 3.5e-9 based on 2017 paper (from drosophila)
            self.mut = Parameter(3.5e-9, 1e-9, 1e-8, "mut")

        # exp (mosquito right now -> change N1/N2/reco/mut for human)
        elif simulator == simulation.exp:
            #self.N1 = Parameter(9000, 1000, 30000, "N1")
            #self.N2 = Parameter(5000, 1000, 30000, "N2")
            self.N1 = Parameter(1e6, 500e3, 3e6, "N1")
            self.N2 = Parameter(1e6, 500e3, 3e6, "N2")
            self.T1 = Parameter(2000, 1500, 5000, "T1")
            self.T2 = Parameter(350, 100, 1500, "T2")
            self.growth = Parameter(0.005, 0.0, 0.05, "growth")
            #self.reco = Parameter(1.25e-8, 1e-9, 1e-7, "reco")
            #self.mut = Parameter(1.25e-8, 1e-9, 1e-7, "mut")

            self.reco = Parameter(8.4e-09, 1e-9, 1e-8, "reco") # stdpopsim
            # 3.5e-9 based on 2017 paper (from drosophila)
            self.mut = Parameter(3.5e-9, 1e-9, 1e-8, "mut")

        # three_epoch (mosquito right now)
        elif simulator == simulation.three_epoch:
            d = 1000
            self.Na = Parameter(384845.04236326, 384845.04236326-d, 384845.04236326+d, "Na")
            self.N1 = Parameter(1891371.2275129908, 1891371.2275129908-d, 1891371.2275129908+d, "N1")
            self.N2 = Parameter(11140821.633397933, 11140821.633397933-d, 11140821.633397933+d, "N2")
            self.T1 = Parameter(60447.09280337712, 60447.09280337712-d, 60447.09280337712+d, "T1")
            self.T2 = Parameter(22708.95299729848, 22708.95299729848-d, 22708.95299729848+d, "T2")

            self.reco = Parameter(8.4e-09, 1e-9, 1e-8, "reco") # stdpopsim
            # 3.5e-9 based on 2017 paper (from drosophila)
            self.mut = Parameter(3.5e-9, 1e-9, 1e-8, "mut")

        elif simulator == simulation.dadi_joint:
            upper_range = 8
            lower_range = 0.1

            # params = CM_vs_UG_dadi_joint = {
            #     'NI': 432139,
            #     'TG': 84723,
            #     'NF': 11040070,
            #     'TS': 3377,
            #     'NI1': 38787744,
            #     'NI2': 24729852,
            #     'NF1': 13116347,
            #     'NF2': 3575499,
            #     'reco': 1.45e-8,
            #     'mut': 3.5e-9
            # }

            # see 2017 paper, supplementary table 2, line 62
            params = BFA_vs_GNB_dadi_joint = {
                'AIC': 33307,
                'NI': 420722,
                'TS': 2247,
                'NI1': 18221645,
                'NI2': 11907698,
                'NF1': 42072233,
                'NF2': 42072201,
                '2NIm': None,  # 'NA' interpreted as None
                'TG': 89511,
                'NF': 9438040
            }
            



            self.NI = Parameter(params['NI'], 5000, 2000000, "NI")
            self.TG = Parameter(params['TG'], 50000, 140000, "TG") # SM: 60k -> 50k
            self.NF = Parameter(params['NF'], params['NF'] * lower_range, params['NF'] * upper_range, "NF")
            self.TS = Parameter(params['TS'], 1000, 40000, "TS") # SM: 10k -> 5k -> 1k, 50k -> 40k
            # strong evidence of recent population expansion in sub-Saharan Africa, thus NI upper range < NF lower range
            # https://academic.oup.com/mbe/article/18/7/1353/992401
            self.NI1 = Parameter(params['NI1'], params['NI1'] * lower_range, params['NI1'] * upper_range, "NI1")
            self.NI2 = Parameter(params['NI2'], params['NI2'] * lower_range, params['NI2'] * upper_range, "NI2")
            self.NF1 = Parameter(params['NF1'], params['NF1'] * lower_range, params['NF1'] * upper_range, "NF1")
            self.NF2 = Parameter(params['NF2'], params['NF2'] * lower_range, params['NF2'] * upper_range, "NF2")

            # stdpopsim
            self.reco = Parameter(1.45e-8, 1e-9, 1e-8, "reco") 
            # 3.5e-9 based on 2017 paper (from drosophila)
            self.mut = Parameter(3.5e-9, 1e-9, 1e-8, "mut")

        
        # mosquito dadi joint mig models: CM_vs_UG (1st line, line 112)
        elif simulator == simulation.dadi_joint_mig:
            upper_range = 8
            lower_range = 0.1

            # params = CM_vs_UG_dadi_joint_mig = {
            #                                 "Population_pair": "CMS_savanna_vs_UGS",
            #                                 "migration_or_no?": "sym_mig",
            #                                 "AIC": 173879,
            #                                 "NI": 367110,
            #                                 "TS": 31783,
            #                                 "NI1": 3098698,
            #                                 "NI2": 219272,
            #                                 "NF1": 8594624,
            #                                 "NF2": 6104686,
            #                                 "2NIm": 4.51427274,
            #                                 "TG": 99659,
            #                                 "NF": 31710315
            #                             }
            
            # see 2017 paper, supplementary table 2, line 72
            params = BFA_vs_GNB_dadi_joint_mig = {
                'AIC': 32780,
                'NI': 416431,
                'TS': 4459,
                'NI1': 6022347,
                'NI2': 3480335,
                'NF1': 19519009,
                'NF2': 41639292,
                '2NIm': 19.99418964,  # This value is given directly
                'TG': 91175,
                'NF': 9086281
            }

            self.NI = Parameter(params['NI'], params['NI'] * lower_range, params['NI'] * upper_range, "NI")
            self.TG = Parameter(params['TG'], 50000, 140000, "TG") # SM: 70k -> 50k
            self.NF = Parameter(params['NF'], params['NF'] * lower_range, params['NF'] * upper_range, "NF")
            self.TS = Parameter(params['TS'], 1000, 40000, "TS") # SM: 10k -> 5k -> 1k, 50k -> 40k
            # strong evidence of recent population expansion, thus NI upper range < NF lower range
            # https://academic.oup.com/mbe/article/18/7/1353/992401
            # agrarian revolution in sub-Saharan Africa approximately 10,000–4,000 years ago could be linked to population expansion of A.Gambiae
            self.NI1 = Parameter(params['NI1'], params['NI1'] * lower_range, params['NI1'] * upper_range, "NI1")
            self.NI2 = Parameter(params['NI2'], params['NI2'] * lower_range, params['NI2'] * upper_range, "NI2")
            self.NF1 = Parameter(params['NF1'], params['NF1'] * lower_range, params['NF1'] * upper_range, "NF1")
            self.NF2 = Parameter(params['NF2'], params['NF2'] * lower_range, params['NF2'] * upper_range, "NF2")
            # 20 as the upper bound of the dadi model
            self.MG = Parameter(params['2NIm'], 0, 100, "MG") # SM: 60 -> 100

            # stdpopsim
            self.reco = Parameter(1.45e-8, 1e-9, 1e-8, "reco") 
            # 3.5e-9 based on 2017 paper (from drosophila)
            self.mut = Parameter(3.5e-9, 1e-9, 1e-8, "mut")


        # im
        elif simulator == simulation.im:
            self.N1 = Parameter(9000, 1000, 30000, "N1")
            self.N2 = Parameter(5000, 1000, 30000, "N2")
            self.N_anc = Parameter(15000, 1000, 25000, "N_anc")
            self.T_split = Parameter(2000, 500, 20000, "T_split")
            self.mig = Parameter(0.05, -0.2, 0.2, "mig")
            self.reco = Parameter(1.25e-8, 1e-9, 1e-7, "reco")
            self.mut = Parameter(1.25e-8, 1e-9, 1e-7, "mut")

        # ooa2
        elif simulator == simulation.ooa2:
            self.N1 = Parameter(9000, 1000, 30000, "N1")
            self.N2 = Parameter(5000, 1000, 30000, "N2")
            self.N3 = Parameter(12000, 1000, 30000, "N3")
            self.N_anc = Parameter(15000, 1000, 25000, "N_anc")
            self.T1 = Parameter(2000, 1500, 5000, "T1")
            self.T2 = Parameter(350, 100, 1500, "T2")
            self.mig = Parameter(0.05, -0.2, 0.2, "mig")
            self.reco = Parameter(1.25e-8, 1e-9, 1e-7, "reco")
            self.mut = Parameter(1.25e-8, 1e-9, 1e-7, "mut")

        # post ooa
        elif simulator == simulation.post_ooa:
            self.N1 = Parameter(9000, 1000, 30000, "N1")
            self.N2 = Parameter(5000, 1000, 30000, "N2")
            self.N3 = Parameter(12000, 1000, 30000, "N3")
            self.N_anc = Parameter(15000, 1000, 25000, "N_anc")
            self.T1 = Parameter(2000, 1500, 5000, "T1")
            self.T2 = Parameter(350, 100, 1500, "T2")
            self.mig = Parameter(0.05, -0.2, 0.2, "mig")
            self.reco = Parameter(1.25e-8, 1e-9, 1e-7, "reco")
            self.mut = Parameter(1.25e-8, 1e-9, 1e-7, "mut")

        # ooa3 (dadi)
        elif simulator == simulation.ooa3:
            self.N_A = Parameter(None, 1000, 30000, "N_A")
            self.N_B = Parameter(2100, 1000, 20000, "N_B")
            self.N_AF = Parameter(12300, 1000, 40000, "N_AF")
            self.N_EU0 = Parameter(1000, 100, 20000, "N_EU0")
            self.N_AS0 = Parameter(510, 100, 20000, "N_AS0")
            self.r_EU = Parameter(0.004, 0.0, 0.05, "r_EU")
            self.r_AS = Parameter(0.0055, 0.0, 0.05, "r_AS")
            self.T_AF = Parameter(8800, 8000, 15000, "T_AF")
            self.T_B = Parameter(5600, 2000, 8000, "T_B")
            self.T_EU_AS = Parameter(848, 100, 2000, "T_EU_AS")
            self.m_AF_B = Parameter(25e-5, 0.0, 0.01, "m_AF_B")
            self.m_AF_EU = Parameter(3e-5, 0.0,  0.01, "m_AF_EU")
            self.m_AF_AS = Parameter(1.9e-5, 0.0, 0.01, "m_AF_AS")
            self.m_EU_AS = Parameter(9.6e-5, 0.0, 0.01, "m_EU_AS")
            self.reco = Parameter(1.25e-8, 1e-9, 1e-7, "reco")
            self.mut = Parameter(1.25e-8, 1e-9, 1e-7, "mut")

            '''
            # additional params for OOA (dadi) TODO where did these come from?
            self.m_AF_B = Parameter(None, 0, 50, "m_AF_B")   # e-5
            self.m_AF_EU = Parameter(None, 0, 50, "m_AF_EU") # e-5
            self.m_AF_AS = Parameter(None, 0, 50, "m_AF_AS") # e-5
            self.m_EU_AS = Parameter(None, 0, 50, "m_EU_AS") # e-5
            '''

        else:
            sys.exit(str(simulator) + " not supported")

    def update(self, names, values):
        """Based on generator proposal, update desired param values"""
        assert len(names) == len(values), (names, values)

        for j in range(len(names)):
            param = names[j]

            # credit: Alex Pan (https://github.com/apanana/pg-gan)
            attr = getattr(self, param)
            if attr is None:
                sys.exit(param + " is not a recognized parameter.")
            else:
                attr.value = values[j]
