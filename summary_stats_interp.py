"""
Compare summary statistics from real data with data simulated under the
inferred parameters.
Author: Sara Mathieson, Rebecca Riley
Date: 1/27/23
"""

# python imports
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import pickle
from scipy.special import rel_entr
import pandas as pd
import tensorflow as tf




# our imports
import global_vars
import ss_helpers
import util

# globals
NUM_TRIAL = 1
# statistic names
NAMES = [
    "minor allele count (SFS)",
    "inter-SNP distances",
    "distance between SNPs",
    "Tajima's D",
    r'pairwise heterozygosity ($\pi$)',
    "number of haplotypes",
    "Hudson's Fst"]
FST_COLOR = "purple"

# for ooa2 (YRI/CEU) (no longer supported)
#FSC_PARAMS = [21017, 0.0341901, 3105.5, 21954, 33077.5, 2844, 1042]

DADI = True

# for dadi (mosquito, BFS population)
#DADI_NAMES = ["Na", "N1", "N2", "T1", "T2"]
#DADI_PARAMS = [384845.04236326, 1891371.2275129908, 11140821.633397933, 60447.09280337712, 22708.95299729848]

DADI_NAMES = ["NI", "TG", "NF", "TS", "NI1", "NI2", "NF1", "NF2"]
DADI_PARAMS = [420646, 89506, 9440437, 2245, 18328570, 42062652, 42064645, 42064198]
DADI_MIG_PARAMS = [415254, 93341, 8292759, 11637, 2635696, 2748423, 11101754, 11439976, 40]

def main():
    # in_file_data_1 = {'model': 'dadi_joint', 'params': 'NI,TG,NF,TS,NI1,NI2,NF1,NF2', 'data_h5': None, 'bed_file': None, 'reco_folder': None, 'grid': None, 'toy': None, 'seed': 1833, 'sample_sizes': '98,98', 'param_values': None}
    # #in_file_data_1 = {'model': 'dadi_joint_mig', 'params': 'NI,TG,NF,TS,NI1,NI2,NF1,NF2,MG', 'data_h5': None, 'bed_file': None, 'reco_folder': None, 'grid': None, 'toy': None, 'seed': 1833, 'sample_sizes': '98,98', 'param_values': None}
    # in_file_data_2 = {'model': 'dadi_joint_mig', 'params': 'NI,TG,NF,TS,NI1,NI2,NF1,NF2,MG', 'data_h5': None, 'bed_file': None, 'reco_folder': None, 'grid': None, 'toy': None, 'seed': 1833, 'sample_sizes': '98,98', 'param_values': None}
    # #in_file_data_3 = {'model': 'dadi_joint_mig', 'params': 'NI,TG,NF,TS,NI1,NI2,NF1,NF2,MG', 'data_h5': None, 'bed_file': None, 'reco_folder': None, 'grid': None, 'toy': None, 'seed': 1833, 'sample_sizes': '98,98', 'param_values': None}
    # print(in_file_data_1)
    # print(in_file_data_2)
    # param_values_1 = DADI_PARAMS = [420646, 89506, 9440437, 2245, 18328570, 42062652, 42064645, 42064198]
    # param_values_2 = DADI_PARAMS = [420646, 89506, 9440437, 2245, 18328570, 42062652, 42064645, 42064198]
    # param_values_2 = DADI_MIG_PARAMS = [415254, 93341, 8292759, 11637, 2635696, 2748423, 11101754, 11439976, 60]
    # opts_1, posterior_1 = util.parse_args(in_file_data = in_file_data_1, param_values=param_values_1)
    # opts_2, posterior_2 = util.parse_args(in_file_data = in_file_data_2, param_values=param_values_2)

    # generator_1, iterator_1, parameters_1, sample_sizes_1 = util.process_opts(opts_1)
    # generator_1.update_params(posterior_1)
    
    # generator_2, iterator_2, parameters_2, sample_sizes_2 = util.process_opts(opts_2)
    # generator_2.update_params(posterior_2)
    
    # real_matrices_baseline = generator_1.simulate_batch(batch_size=1, neg1=False)
    # print("finish real baseline")
    # real_matrices_region_baseline = generator_1.simulate_batch(batch_size=NUM_TRIAL, neg1=False,
    #         region_len=True)
    # print("finish real region_len baseline")
    
    # '''
    # NOTE: for summary stats, use neg1=False to keep hap data as 0/1 (not -1/1)
    # NOTE: use region_len=True for Tajima's D (i.e. not all regions have same S)
    # '''
    
    # print("loading pretrained model")
    # discriminator = tf.keras.models.load_model("./pretrained_model/GNB-model_compare/disc")
    # #discriminator.dense3 = tf.keras.layers.Dense(2)
    
    # ss_distances_pd = pd.DataFrame(columns=['pop_sfs_1','pop_dist_1','pop_ld_1','Tajimas D_1','pi_1','no.haps_1', 'pop_sfs_2','pop_dist_2','pop_ld_2','Tajimas D_2','pi_2','no.haps_2', 'FST', 'prob_0', 'prob_1'])

    
    # for index, samples in enumerate(range(500)):
    #     # real
    #     # real_matrices = generator_1.simulate_batch(batch_size=NUM_TRIAL, neg1=False)
    #     # print("finish real")
    #     # real_matrices_region = generator_1.simulate_batch(batch_size=NUM_TRIAL, neg1=False,
    #     #     region_len=True)
    #     # print("finish real region_len")
    #     # sim
    #     sim_matrices = generator_2.simulate_batch(batch_size=NUM_TRIAL, neg1=False)
    #     print("finish sim")
    #     sim_matrices_region = generator_2.simulate_batch(batch_size=NUM_TRIAL,
    #         neg1=False, region_len=True)
    #     print("finish sim region_len")
        
    
    #     # real_matrices_logits = discriminator(real_matrices_baseline, training=False)
    #     # real_matrices_prob = (tf.nn.sigmoid(real_matrices_logits)).numpy()
        
    #     sim_matrices_logits = discriminator(sim_matrices, training=False)
    #     sim_matrices_prob = (tf.nn.sigmoid(sim_matrices_logits)).numpy().tolist()

    #     num_pop = len(sample_sizes_1)

    #     # one pop models
    #     if num_pop == 1:
    #         nrows, ncols = 3, 2
    #         size = (7, 7)
    #         first_pop, second_pop = [], []

    #     # two pop models
    #     elif num_pop == 2:
    #         nrows, ncols = 4, 4
    #         size = (14, 10)
    #         first_pop, second_pop = [0], [1]

    #     # OOA3
    #     elif opts_1.model in ['ooa3']:
    #         nrows, ncols = 6, 4
    #         size = (14, 14)
    #         first_pop, second_pop = [0, 0, 1], [1, 2, 2]

    #     else:
    #         print("unsupported number of pops", num_pop)

    #     # split into individual pops
    #     real_all, real_region_all, sim_all, sim_region_all = \
    #         split_matrices(real_matrices_baseline, real_matrices_baseline, sim_matrices,
    #         sim_matrices_region, sample_sizes_1)

    #     # stats for all populations
    #     real_stats_lst = []
    #     sim_stats_lst = []
    #     for p in range(num_pop):
    #         print("real stats for pop", p)
    #         #temp solution for fixing difference in coalescent tree length
    #         # if "sim" in input_file:
    #         #     print("reading sim file")
    #         #     real_stats_pop = ss_helpers.stats_all(real_all[p], real_region_all[p], 5000000)
    #         # else:
    #         real_stats_pop = ss_helpers.stats_all(real_all[p], real_region_all[p])
    #         print("sim stats for pop", p)
    #         sim_stats_pop = ss_helpers.stats_all(sim_all[p], sim_region_all[p])

    #         real_stats_lst.append(real_stats_pop)
    #         sim_stats_lst.append(sim_stats_pop)

    #     print("got through main stats")

    #     # Fst over all pairs
    #     real_fst_lst = []
    #     sim_fst_lst = []
    #     for pi in range(len(first_pop)):
    #         a = first_pop[pi]
    #         b = second_pop[pi]
    #         real_ab = np.concatenate((np.array(real_all[a]), np.array(real_all[b])),
    #             axis=1)
    #         sim_ab = np.concatenate((np.array(sim_all[a]), np.array(sim_all[b])),
    #             axis=1)

    #         # compute Fst
    #         real_fst = ss_helpers.fst_all(real_ab)
    #         sim_fst = ss_helpers.fst_all(sim_ab)
    #         real_fst_lst.append(real_fst)
    #         sim_fst_lst.append(sim_fst)

    #     print("got through fst")
      
      
    #     #stat_lst in each pop: [[pop_sfs (no. sfs bar)], [pop_dist (no.snps * NUM_TRIAL)], 
    #     # [pop_ld (NUM_LD)], [Tajima's D (NUM_TRIAL)], [pi (NUM_TRIAL)], [num haps(NUM_TRIAL)]]
    #     #fst_lst in each pop:[FST (NUM_TRIAL)]
    
    #     distances = process_ss_datas(real_stats_lst, sim_stats_lst, real_fst_lst, sim_fst_lst)
    #     # ss_distances_pd["real_0_prob", "real_1_prob"].loc[index] = real_matrices_prob
    #     print(distances, sim_matrices_prob)
    #     ss_distances_pd.loc[index] = distances + sim_matrices_prob[0]
        
    # print(ss_distances_pd)
    # ss_distances_pd.to_csv("/data/SBBS-FumagalliLab/mosquito_gan/pg-gan-mosquito/sim_out/ss_distances_pd.csv", mode="a", index=False, header=True)
    
    ss_distances_pd = pd.read_csv("/data/SBBS-FumagalliLab/mosquito_gan/pg-gan-mosquito/sim_out/ss_distances_pd.csv", header = 0)
    print(ss_distances_pd)
    
    #########################################################################################################
    #Correlation heatmap
    #########################################################################################################    
    X = ss_distances_pd.iloc[:,:-2]
    y = ss_distances_pd.iloc[:,-1:]
    color  = np.where(ss_distances_pd['prob_0'] > ss_distances_pd['prob_1'], '0', '1')
    print(color)
    
    correlation = X.corr()
    plt.figure(figsize=(16, 5))
    heatmap = sns.heatmap(correlation, annot=True, linewidths=0, vmin=-0.75, vmax=0.75, cmap="RdBu_r")
    fig = heatmap.get_figure()
    fig.savefig("/data/SBBS-FumagalliLab/mosquito_gan/pg-gan-mosquito/sim_out/heatmap.png")
    plt.clf()

    #########################################################################################################
    #PCA
    #########################################################################################################
    from sklearn import datasets, decomposition
    pca = decomposition.PCA(n_components=2)
    pca.fit(X)
    pca_X = pca.transform(X)
    print(pca.explained_variance_ratio_)
    
    plt.figure(figsize=(16, 5))
    plot = plt.scatter(pca_X[:,0], pca_X[:,1], c=ss_distances_pd['prob_1'])
    plt.legend(handles=plot.legend_elements()[0])
    plt.savefig("/data/SBBS-FumagalliLab/mosquito_gan/pg-gan-mosquito/sim_out/pca.png") 

    #########################################################################################################
    #RFE for regression
    #########################################################################################################
    # explore the algorithm wrapped by RFE
    # from numpy import mean
    # from numpy import std
    # from sklearn.datasets import make_classification
    # from sklearn.model_selection import cross_val_score
    # from sklearn.model_selection import RepeatedStratifiedKFold
    # from sklearn.feature_selection import RFE
    # from sklearn.linear_model import LogisticRegression
    # from sklearn.linear_model import Perceptron
    # from sklearn.tree import DecisionTreeClassifier
    # from sklearn.ensemble import RandomForestClassifier
    # from sklearn.ensemble import GradientBoostingClassifier
    # from sklearn.pipeline import Pipeline
    # from matplotlib import pyplot
    
    
    # def get_models():
    #     models = dict()
    #     # lr
    #     rfe = RFE(estimator=LogisticRegression(), n_features_to_select=5)
    #     model = DecisionTreeClassifier()
    #     models['lr'] = Pipeline(steps=[('s',rfe),('m',model)])
    #     # cart
    #     rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=5)
    #     model = DecisionTreeClassifier()
    #     models['cart'] = Pipeline(steps=[('s',rfe),('m',model)])
    #     # rf
    #     rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=5)
    #     model = DecisionTreeClassifier()
    #     models['rf'] = Pipeline(steps=[('s',rfe),('m',model)])
    #     # gbm
    #     rfe = RFE(estimator=GradientBoostingClassifier(), n_features_to_select=5)
    #     model = DecisionTreeClassifier()
    #     models['gbm'] = Pipeline(steps=[('s',rfe),('m',model)])
    #     return models

    # # evaluate a give model using cross-validation
    # def evaluate_model(model, X, y):
    #     cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    #     scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    #     return scores

    # # define dataset
    # # get the models to evaluate
    # models = get_models()
    # # evaluate the models and store results
    # results, names = list(), list()
    # for name, model in models.items():
    #     scores = evaluate_model(model, X, y)
    #     results.append(scores)
    #     names.append(name)
    #     print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
    # # # plot model performance for comparison
    # # pyplot.boxplot(results, labels=names, showmeans=True)
    # # pyplot.show()
    
    
    


def process_ss_datas(real_stats_pop_lst, sim_stats_pop_lst, real_fst_lst, sim_fst_lst):
    distances = []
    
    for index in range(2):
        real_stats_lst, sim_stats_lst = real_stats_pop_lst[index], sim_stats_pop_lst[index]
        real, sim = real_stats_lst[0], sim_stats_lst[0]
        num_sfs = len(real)
        real_sfs = [sum(rs)/num_sfs for rs in real]
        sim_sfs = [sum(ss)/num_sfs for ss in sim]
        sfs_dist = ss_helpers.calc_distribution_dist(real_sfs, sim_sfs)
        distances.append(sfs_dist)

        for index in range(2):
            real, sim = real_stats_lst[index], sim_stats_lst[index]    
            nbin = ss_helpers.NUM_LD
            max_dist = 500 # TODO make flexible! (5k for mosquito, 20k for human)
            dist_bins = np.linspace(0,max_dist,nbin)
            real_mean = [np.mean(rs) for rs in real]
            sim_mean = [np.mean(ss) for ss in sim]
            real_stddev = [np.std(rs) for rs in real]
            sim_stddev = [np.std(ss) for ss in sim]
            LD_dist = ss_helpers.calc_distribution_dist(real_mean, sim_mean)
            distances.append(LD_dist)

        for index, (real, sim) in enumerate(zip(real_stats_lst[3:], sim_stats_lst[3:])):
            # if any(n < 0 for n in real) or any(n < 0 for n in sim):
            #     real = list(np.asarray(real) + 5)
            #     sim = list(np.asarray(sim) + 5)
            dist = ss_helpers.calc_distribution_dist(real, sim)
            distances.append(dist)

        # if any(n < 0 for n in real_fst_lst[0]) or any(n < 0 for n in sim_fst_lst[0]):
        #     print(real_fst_lst[0], sim_fst_lst[0])
        #     real_fst_lst[0] = list(np.asarray(real_fst_lst[0]) + 5)
        #     sim_fst_lst[0] = list(np.asarray(sim_fst_lst[0]) + 5)
    fst_dist = ss_helpers.calc_distribution_dist(real_fst_lst[0], sim_fst_lst[0])

    distances.append(fst_dist)
        
    return distances    
    
    
            

            
            
    
    
    
    # finall plotting call
    # plot_stats_all(nrows, ncols, size, real_stats_lst, sim_stats_lst,
    #     real_fst_lst, sim_fst_lst, output_file)

def split_matrices(real_matrices, real_matrices_region, sim_matrices,
    sim_matrices_region, sample_sizes):

    # set up empty arrays
    real_all, real_region_all, sim_all, sim_region_all = [], [], [], []

    start_idx = 0
    for s in sample_sizes:
        end_idx = start_idx + s

        # parse real matrices
        real_p = real_matrices[:,start_idx:end_idx,:,:]
        real_region_p = []
        for item in real_matrices_region:
            real_region_p.append(item[start_idx:end_idx,:,:])
        real_all.append(real_p)
        real_region_all.append(real_region_p)

        # parse sim matrices
        sim_p = sim_matrices[:,start_idx:end_idx,:,:]
        sim_region_p = []
        for item in sim_matrices_region:
            sim_region_p.append(item[start_idx:end_idx,:,:])
        sim_all.append(sim_p)
        sim_region_all.append(sim_region_p)

        # last step: update start_idx
        start_idx = end_idx

    return real_all, real_region_all, sim_all, sim_region_all

# one, two, and three pops
def plot_stats_all(nrows, ncols, size, real_stats_lst, sim_stats_lst,
    real_fst_lst, sim_fst_lst, output, title_data=None):
    num_pop = len(real_stats_lst)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=size)

    if title_data is not None:
        fig.suptitle(title_data["title"], fontsize=title_data["size"])
    
    # labels and colors
    labels = global_vars.SS_LABELS[:num_pop]
    sim_label = global_vars.SS_LABELS[-1]
    colors = global_vars.SS_COLORS[:num_pop]
    sim_color = global_vars.SS_COLORS[-1]

    # plot each population
    rows = [0, 0, 3]
    cols = [0, 2, 2]
    single = True if num_pop == 1 else False
    for p in range(num_pop): # one/two pop won't use last indices
        real_color = colors[p]
        real_label = labels[p]
        real_pop = real_stats_lst[p]
        sim_pop = sim_stats_lst[p]
        plot_population(axes, rows[p], cols[p], real_color, real_label,
            real_pop, sim_color, sim_label, sim_pop, single=single)

    # Fst (all pairs)
    cidx = 0
    first_pop = [0, 0, 1]
    second_pop = [1, 2, 2]
    if num_pop == 2:
        cidx = 1
    for pi in range(len(real_fst_lst)): # pi -> pair index
        pair_label = labels[first_pop[pi]] + "/" + labels[second_pop[pi]]
        ss_helpers.plot_generic(axes[3+pi][cidx], NAMES[6], real_fst_lst[pi],
            sim_fst_lst[pi], FST_COLOR, sim_color, pop=pair_label,
            sim_label=sim_label)

    # overall legend
    if num_pop >= 2:
        for p in range(num_pop):
            p_real = mpatches.Patch(color=colors[p], label=labels[p] + \
                ' real data')
            p_sim = mpatches.Patch(color=sim_color, label=labels[p] + \
                ' sim data')
            if num_pop == 2:
                axes[3][0+3*p].axis('off')
                axes[3][0+3*p].legend(handles=[p_real, p_sim], loc=10,
                    prop={'size': 18})
            if num_pop == 3:
                axes[3+p][1].axis('off')
                axes[3+p][1].legend(handles=[p_real, p_sim], loc=10,
                    prop={'size': 18})

    if num_pop == 2:
        axes[3][2].axis('off')

    plt.tight_layout()

    if output != None:
        plt.savefig(output, dpi=350)
    else:
        plt.show()

def plot_population(axes, i, j, real_color, real_label, real_tuple, sim_color,
    sim_label, sim_tuple, single=False):
    """
    Plot all 6 stats for a single population, starting from the (i,j) subplot.
    """
    for r in range(3):
        for c in range(2):
            idx = 2*r+c
            ss_helpers.plot_generic(axes[i+r][j+c], NAMES[idx], real_tuple[idx],
                sim_tuple[idx], real_color, sim_color, pop=real_label,
                sim_label=sim_label, single=single)

# only called once per summary_stats call
def get_title_from_trial_data(opts, param_values, sample_sizes):
    num_pops = len(sample_sizes)
    if num_pops == 1:
        FONT_SIZE = 8
        CHAR_LIMIT = 90
    else:
        FONT_SIZE = 12
        CHAR_LIMIT = 130

    # helper functions ------------------------------------------------------
    def fix_value_length(prefix, value):
        value = prefix + str(value)
        len_value = len(value)

        if len_value <= CHAR_LIMIT:
            return str(value)
        
        num_split = len_value // CHAR_LIMIT + 1
        index_width = len(value)// num_split # should work for strs or lists

        from_index = 0
        to_index = index_width

        values_fixed = ""
        for n in range(num_split - 1):
            values_fixed = values_fixed + str(value[from_index:to_index]) + "\n"
            from_index = from_index + index_width
            to_index = to_index + index_width
            
        # no "/n" on the last one
        values_fixed = values_fixed + str(value[from_index:to_index])
    
        return values_fixed

    def round_params(param_list):
        # cast to floats        
        for i in range(len(param_list)):
            if abs(float(param_list[i])) < 1.0:
                param_list[i] = round(param_list[i], 6)
            else:
                param_list[i] = int(param_list[i])

        return param_list
    # -----------------------------------------------------------
    
    params_using = param_values.copy() if opts.param_values is None else opts.param_values.copy()
    params_using = round_params(params_using)
    
    if opts.data_h5 is None and opts.bed is None and opts.reco_folder is None:
        s_source = "data_h5: None, bed: None, reco: None"
    else:
        s_source = "data_h5: "+str(opts.data_h5)+",\nbed: "+opts.bed+\
                   ",\nreco: "+opts.reco_folder
    
    s_model = "model: " + opts.model + ", "
    s_ss = "sample_sizes: " + str(sample_sizes) + ", "
    s_seed = "seed: " + str(opts.seed) + ", "
    s_num_trial = "SSTATS_TRIALS: " + str(NUM_TRIAL) + ", "
    s_params = "params: " + opts.params + ", "
    s_param_values = fix_value_length("param_values: ", params_using)+","
    
    title = s_num_trial + s_model + s_ss + s_seed + "\n" + s_params + "\n" +\
        s_param_values + "\n" + s_source + "\n"

    return {"size": FONT_SIZE, "title": title}

main()
