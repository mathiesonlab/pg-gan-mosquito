import numpy as np
import pandas as pd
import itertools

import seaborn as sns
import random
import statistics
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn import preprocessing
from sklearn.decomposition import PCA
import umap.umap_ as umap
import ot
from scipy import stats

    
def run_pca(X, ncomp = 0.95):
    '''
    perform pca decomposition on X

    input:
    X: numpy ndarray containing genotypic data [sample size, SNPs * batchsize]
    ncomp: If 0 < n_components < 1 and svd_solver == 'full', select the number
     of components such that the amount of variance that needs to be explained
     is greater than the percentage specified by n_components
    output:
    PCs: numpy ndarray of the first ncomp components [sample size, ncomp]
    '''
    scaler = preprocessing.StandardScaler()
    data_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=ncomp)
    PCs = pca.fit_transform(data_scaled)
    return pca, PCs

def run_umap(X, ncomp = 2):
    '''
    perform umap decomposition on X

    input:
    X: numpy ndarray containing genotypic data [sample size, SNPs * batchsize]
    output:
    PCs: numpy ndarray of the first ncomp components [sample size, ncomp]
    '''
    # Instantiate a UMAP object with two output dimensions
    umap_obj = umap.UMAP(n_components=ncomp)
    # Fit the UMAP object to the data
    X_umap = umap_obj.fit_transform(X)
    return X_umap


def wasserstein_dis_2d(pc1, pc2, reg=1e-3):
    '''
    calculate 2d wasserswtein_dis

    inspired from 
https://gitlab.inria.fr/ml_genetics/public/artificial_genomes/-/blob/master/plotting_notebooks/short/plot_utils.py

    input: pc1, pc2: numpy ndarrays of [sample size, 2], 2 being the first 2 prinicipal components of data
    output: sc: float, Optimal transportation loss to transfer one distribution to another
    '''
    #n = number of samples
    n = pc1.shape[0]
    #  a and b are source and target weights (histograms, both sum to 1)
    #here assumed uniform distribution
    a, b = np.ones((n,)) / n, np.ones((n,)) / n
    # M is the (pc1, pc2) metric cost matrix
    M = ot.dist(pc1, pc2)
    M /= M.max()
    sc = ot.sinkhorn2(a, b, M, reg)
    return sc

def genotype_data_eval(real_all, sim_all, batch = 50):
    """
    wrapper for comparing fixed snp genotype data from simulate_batch() and real_batch()

    input: 
    real_all: fixed snp genotype data from real data
    sim_all: fixed snp genotype data derived from simulated msprime params
    batch: number of batches to consider

    output: avg_2d_wasserstein_dis: float, average 2d wasserstein distance across batches, used for hyperparameter tuning evaluation metric

    """
    assert real_all.shape == sim_all, "real and simulated genotype data not same shape"
    #TODO: add flexibility when considering 1 pop
    assert batch <= real_all.shape[1], "batch larger than no. batches in genotype data"
    temp_umap_comparison, batch_scores = [],[]
    for batch in random.sample(range(0, 500), batches):
        for genotype_array in [real_all, sim_all]:
            #currently only tested for two pop model
            batch_array = genotype_array[:,batch,:, :, 0]
            #concatenate the populations
            batch_array = batch_array.reshape(-1,batch_array.shape[-1])
            pca, pc = run_pca(batch_array)
            umap_reduced = run_umap(pc)
            temp_umap_comparison.append(umap_reduced)

        umap_real, umap_sim = temp_umap_comparison[0], temp_umap_comparison[1]
        score = wasserstein_dis_2d(umap_real, umap_sim, reg = 0.01)
        batch_scores.append(score)
    avg_2d_wasserstein_dis = float(statistics.mean(batch_scores))

    return avg_2d_wasserstein_dis





if __name__ == "__main__":
    #declare path and globals 
    mypath = "./sim_data/"
    files = ['dadi_joint_param_real.npy', 'dadi_joint_param_sim.npy', 'dadi_joint_sim+10.npy', 'dadi_joint_sim+25.npy', 'dadi_joint_sim+40.npy']
    #files = ['dadi_joint_param_real.npy', 'dadi_joint_param_sim.npy', 'dadi_joint_param_sim-10.npy', 'dadi_joint_param_sim-25.npy', 'dadi_joint_param_sim-40.npy']
    pairwise = itertools.combinations(files, 2)
    filenames = [file.split('.')[0].split('_')[-1] for file in files]
    arrays = [np.load(mypath + file) for file in files]
    pops = ['GNB', 'BFA']
    batches = 50


    ##############################################################
    #consider data on multiple pca / umap
    ##############################################################
    files_pair = [[files[0], files[index]] for index in range(1, len(files))]
    print(files_pair)
    f_oneway_score_list = []
    for pair in files_pair:
        print(pair)
        array = [np.load(mypath + file) for file in pair]
        print("avg 2d score = ", genotype_data_eval(array[0], array[1]))
        



        
        
    #     print("avg_disparity")
    #     print(pd.DataFrame(avg_disparity).describe())
    #     print("avg_score")
    #     print(pd.DataFrame(avg_score).describe())
    #     print("avg_rotated_score")
    #     print(pd.DataFrame(avg_rotated_score).describe())

    #     f_oneway_score_list.append(avg_score)

    # res = stats.f_oneway(*f_oneway_score_list)
    # print(res)

    # res = stats.tukey_hsd(*f_oneway_score_list)
    # print(res)

    




    # fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    # plt.subplots_adjust(hspace=0.5)
    # #compare similarity between real and simulated umap reduced datasets
    # real_df = final_df.loc[final_df['type'] == 'real']
    # real_umap_values = real_df[['umap_1', 'umap_2']].to_numpy()
    # for ax, sim_type in zip(axs.ravel(), ['sim', 'sim10', 'sim25', 'sim40']):
    #     print("comparing real vs ", sim_type)
    #     sim_df = final_df.loc[final_df['type'] == sim_type]
    #     sim_umap_values = sim_df[['umap_1', 'umap_2']].to_numpy()

    #     mtx1, mtx2, disparity = procrustes(real_umap_values, sim_umap_values)
    #     print("disparity = ", disparity)
    #     score = wasserstein_dis_2d(real_umap_values, sim_umap_values, reg = 0.01)
    #     print("2d-wasserstein distance = ", score)
    #     rotated_score = wasserstein_dis_2d(mtx1, mtx2, reg = 0.01)
    #     print("rotated_2d-wasserstein distance = ", rotated_score)

    #     temp_mtx = np.vstack((mtx1, mtx2))
    #     temp_df = pd.concat([real_df['labels_type'], sim_df['labels_type']]).reset_index()
    #     temp_df['1st'], temp_df['2nd'] = temp_mtx[:,0], temp_mtx[:,1]

    #     sns.scatterplot(x="2nd", y="1st",s=100,hue='labels_type', data=temp_df, ax = ax)
    #     ax.set_ylabel('rotated_umap_1')
    #     ax.set_xlabel('rotated_umap_2')
        

    # plt.savefig(mypath + "umap_comparisons_sep.png")

    # data_types = [' ', 'real']
    # for data_type in data_types:
    #     plt.clf()
    #     scatter = sns.scatterplot(y = 'umap_1', x = 'umap_2', data=subset_df, hue=subset_df['labels'])
    #     plt.ylabel('umap_1')
    #     plt.xlabel('umap_2')
    #     plt.savefig(mypath +  data_type + "_umap.png")




    
                
            
            








