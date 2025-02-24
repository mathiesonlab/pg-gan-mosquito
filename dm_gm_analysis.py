import numpy as np
import pandas as pd
import itertools
from os import listdir
from os.path import isfile, join

import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn import preprocessing
from sklearn.decomposition import PCA
import umap.umap_ as umap
from scipy.spatial import procrustes
import ot


def flatten(matrix_4d):
    '''
    convert the simulated batch of [batchsize, sample size, SNPs, 2] to
    [sample size, SNPs] in preperation for dimensionality reduction
    '''
    #first channel are genotype data, second is snp distance
    matrix_3d = np.squeeze(matrix_4d[:,:,:,0])
    #concat batches into [sample size * batchsize, snps]
    return np.hstack(matrix_3d)
    
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
    print("normalizing raw data")
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


    # mtx1, mtx2, disparity = procrustes(a, b)
    # return mtx1, mtx2, disparity

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


if __name__ == "__main__":

    loss_list = []
    mypath = "./sim_data/"
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    files = list(filter(lambda k: '.npy' in k, files))
    print(files)
    pops = ['GNB', 'BFA']


    ##############################################################
    #consider data on single pca / umap
    ##############################################################
    pc_df_list = []
    umap_np_list = []
    filename_list = []
    for index, file in enumerate(files):
        label_list = []
        type_list = []
        genotype_2pop_list = []
        array = np.load(mypath + file)
        filename = file.split(".")[0].split("_")[-1]
        filename_list.append(filename)
        print("working for ", filename, " data")
        genotype_2pop = []
        for index, pop_name in enumerate(pops):
            genotype_pop = array[index,:,:,:]
            genotype_pop = flatten(genotype_pop)
            genotype_2pop.append(genotype_pop)
            # * sample size
            label_list.extend([pop_name] * 98)
            type_list.extend([filename] * 98)
        genotype_2pop = np.vstack(tuple(genotype_2pop))
        df = pd.DataFrame(genotype_2pop)
        pca, pc = run_pca(genotype_2pop)
        exp_var_pca = pca.explained_variance_ratio_

        pc_df = pd.DataFrame(pc[:,:2], columns = ['pca_1', 'pca_2'])
        pc_df['labels'] = label_list
        pc_df['type'] = type_list
        
        pc_df_list.append(pc_df)
        umap_np_list.append(pc)

    umap_pcs = np.vstack(tuple(umap_np_list))
    print("pca shape", umap_pcs.shape)
    umap_pcs_reduced = run_umap(umap_pcs)
    print("reduced umap shape", umap_pcs_reduced.shape)
    umap_pcs_reduced_df = pd.DataFrame(umap_pcs_reduced, columns = ['umap_1', 'umap_2'])

    final_df = pd.concat(pc_df_list, axis=0, ignore_index = True)
    final_df = pd.concat([final_df, umap_pcs_reduced_df], axis=1)
    print("unique labels:", final_df['labels'].unique())
    print("unique types:", final_df['type'].unique())
    final_df['labels_type'] = list(zip(final_df.labels, final_df.type))
    print(print("unique label_types:", final_df['labels_type'].unique()))
    print("final_df shape", final_df.shape)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    plt.subplots_adjust(hspace=0.5)
    #compare similarity between real and simulated umap reduced datasets
    real_df = final_df.loc[final_df['type'] == 'real']
    real_umap_values = real_df[['umap_1', 'umap_2']].to_numpy()
    for ax, sim_type in zip(axs.ravel(), ['sim', '10', '25', '40']):
        print("comparing real vs ", sim_type)
        sim_df = final_df.loc[final_df['type'] == sim_type]
        sim_umap_values = sim_df[['umap_1', 'umap_2']].to_numpy()

        mtx1, mtx2, disparity = procrustes(real_umap_values, sim_umap_values)
        print("disparity = ", disparity)
        score = wasserstein_dis_2d(real_umap_values, sim_umap_values, reg = 0.01)
        print("2d-wasserstein distance = ", score)
        rotated_score = wasserstein_dis_2d(mtx1, mtx2, reg = 0.01)
        print("rotated_2d-wasserstein distance = ", rotated_score)

        temp_mtx = np.vstack((mtx1, mtx2))
        temp_df = pd.concat([real_df['labels_type'], sim_df['labels_type']]).reset_index()
        temp_df['1st'], temp_df['2nd'] = temp_mtx[:,0], temp_mtx[:,1]

        sns.scatterplot(x="2nd", y="1st",s=100,hue='labels_type', data=temp_df, ax = ax)
        ax.set_ylabel('rotated_umap_1')
        ax.set_xlabel('rotated_umap_2')

    plt.savefig(mypath + "umap_comparisons.png")

    # data_types = [' ', 'real']
    # for data_type in data_types:
    #     plt.clf()
    #     scatter = sns.scatterplot(y = 'umap_1', x = 'umap_2', data=subset_df, hue=subset_df['labels'])
    #     plt.ylabel('umap_1')
    #     plt.xlabel('umap_2')
    #     plt.savefig(mypath +  data_type + "_umap.png")

    
    # pairwise = itertools.combinations(final_df['labels_type'].unique(), 2)
    # for method in ['pca', 'umap']:
    #     for label_1, label_2 in pairwise:
    #         print("working on ", label_1, label_2)
    #         col_name_1 = method + '_1'
    #         col_name_2 = method + '_2'
    #         data_1 = final_df.loc[final_df['labels_type'] == label_1, [col_name_1, col_name_2]].to_numpy()
    #         data_2 = final_df.loc[final_df['labels_type'] == label_2, [col_name_1, col_name_2]].to_numpy()
    #         score = wasserstein_dis_2d(data_1, data_2, reg = 0.01)
    #         loss_list.append([method, label_1, label_2, score])

    # for pairs in loss_list:
    #     print(pairs)




    
                
            
            








