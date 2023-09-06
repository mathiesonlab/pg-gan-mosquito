"""
Template code to draw demographic history plots
Author: Sara Mathieson, Jacky Siu Pui Chung
Date: 09/08/23
"""

import demes
import demesdraw
import simulation
import param_set
import math
import msprime
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    dadi_simulator = getattr(simulation, "dadi_joint_mig")
    params = param_set.ParamSet(simulator = dadi_simulator)
    
    gen_per_year = 11
    
    # described past -> present
    NI = params.NI.value # the initial ancestral population size
    TG = params.TG.value*gen_per_year # the time of when the ancestral population begins to change in size
    NF = params.NF.value # the final ancestral population size, immediately prior to the split
    TS = params.TS.value*gen_per_year # the time of the split
    NI1 = params.NI1.value # the initial sizes of population 1 and population 2
    NI2 = params.NI2.value
    NF1 = params.NF1.value # the final sizes of these two populations
    NF2 = params.NF2.value
    MG = params.MG.value

    #NI, TG, NF, TS, NI1, NI2, NF1, NF2 = [round(i, 0) for i in [9688.065442541832, 131201.1294558641, 4653468.237258127, 3186.3946089052088, 45206590.227156356, 148245663.78417984, 116660890.39055085, 119300459.64111155]]
    NI, TG, NF, TS, NI1, NI2, NF1, NF2, MG = [round(i, 0) for i in [395364.7811206074, 27256.34424951467, 32311207.64527105, 3073.127217151053, 5167434.226015299, 556928.6361825585, 6546727.245392691, 43175886.54323531, 95.01126693866624]]

    # compute growth rates from the start/end sizes and times
    # negative since backward in time
    g1 = -(1/TS) * math.log(NI1/NF1)
    g2 = -(1/TS) * math.log(NI2/NF2)
    g  = -(1/(TG-TS)) * math.log(NI/NF) # ancestral
    #small m
    MG = MG / 2 / NF

    demography = msprime.Demography()
    demography.add_population(name="POP1", initial_size=NF1)
    demography.add_population(name="POP2", initial_size=NF2)
    demography.add_population(name="ANC", initial_size=NF, initially_active=False)

    demography.add_population_parameters_change(time=0, growth_rate=g1, population="POP1")
    demography.add_population_parameters_change(time=0, growth_rate=g2, population="POP2")

    demography.set_symmetric_migration_rate(populations=["POP1","POP2"], rate=MG)
    
    demography.add_population_split(time=TS, derived=["POP1", "POP2"], ancestral="ANC")
    demography.add_population_parameters_change(time=TS, growth_rate=g, population="ANC")
    demography.add_population_parameters_change(time=TG, growth_rate=0, population="ANC")


    
    
    graph = msprime.Demography.to_demes(demography)
    
    log_time = demesdraw.utils.log_time_heuristic(graph)
    log_size = demesdraw.utils.log_size_heuristic(graph)
    fig, ax = plt.subplots()  # use plt.rcParams["figure.figsize"]
    
    demesdraw.tubes(
        graph,
        ax = ax,
        log_time=log_time,
        title="dadi_joint_mig_posterior"
    )
    
    plt.savefig("./deme_draw/dadi_joint_mig_posterior.png")
    demes.dump(graph, "./deme_draw/dadi_joint_mig_posterior.yaml")
    


