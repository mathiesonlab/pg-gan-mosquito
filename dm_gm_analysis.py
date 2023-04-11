import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
# import umap
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
    
def run_pca(X, ncomp = 2):
    '''
    perform pca decomposition on X

    input:
    X: numpy ndarray containing genotypic data [sample size, SNPs * batchsize]
    output:
    PCs: numpy ndarray of the first ncomp components [sample size, ncomp]
    '''
    data_scaled = pd.DataFrame(preprocessing.scale(X))
    print("normalized")
    pca = PCA(n_components=ncomp)
    PCs = pca.fit_transform(data_scaled)
    return PCs

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



if __name__ == "__main__":
        real = np.load("./sim_data/real.npy")
    sim = np.load("./sim_data/sim.npy")
    #real_region = np.load("./sim_data/real_region.npy")
    #sim_region = np.load("./sim_data/sim_region.npy")

    #testing for first population
    GNB_real_genotype = real[0,:,:,:]
    GNB_sim_genotype = sim[0,:,:,:]

    print("genotype data shape")
    print(GNB_real_genotype.shape)
    print(GNB_sim_genotype.shape)

    GNB_real_genotype_flatten = flatten(GNB_real_genotype)
    GNB_sim_genotype_flatten = flatten(GNB_sim_genotype)
    print("genotype flatten data shape")
    print(GNB_sim_genotype_flatten.shape)
    print(GNB_sim_genotype_flatten.shape)

    GNB_real_pc = run_pca(GNB_real_genotype_flatten)
    GNB_sim_pc = run_pca(GNB_sim_genotype_flatten)

    # GNB_real_umap = run_umap(GNB_real_genotype_flatten)
    # GNB_sim_umap = run_umap(GNB_sim_genotype_flatten)

    print("pca components shape")
    print(GNB_real_pc.shape)
    print(GNB_sim_pc.shape)

    # print("umap components shape")
    # print(GNB_real_umap.shape)
    # print(GNB_sim_umap.shape)

    score = wasserstein_dis_2d(GNB_real_pc, GNB_sim_pc, reg = 0.01)
    print("pca score", score)

    # score = wasserstein_dis_2d(GNB_real_umap, GNB_sim_umap, reg = 0.01)
    # print("umap score", score)

