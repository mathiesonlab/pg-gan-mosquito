"""
Summary stat helpers for computing and plotting summary statistics.
Note: "real" should alwasy be first, followed by simulated.
Author: Sara Mathieson, Rebecca Riley
Date: 1/27/23
"""

# python imports
import allel
import libsequence
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.special import rel_entr, softmax
from scipy.stats import wasserstein_distance


# our imports
import global_vars

# GLOBALS
NUM_SFS = 10
NUM_LD  = 15

################################################################################
# PARSE PG-GAN OUTPUT
################################################################################

def parse_mini_lst(mini_lst):
    return [float(remove_numpy(x.replace("[",'').replace("]",'').replace(",",''))) for x in
        mini_lst]

def remove_numpy(string):
    if "(" in string:
        return string[string.index("(")+1:string.index(")")]
    return string

def add_to_lst(total_lst, mini_lst):
    assert len(total_lst) == len(mini_lst), (mini_lst)
    for i in range(len(total_lst)):
        total_lst[i].append(mini_lst[i])

def parse_output(filename, return_acc=False):
    """Parse pg-gan output to find the inferred parameters"""

    def clean_param_tkn(s):
        if s == 'None,':
            return None # this is a common result (not an edge case)

        if s[:-1].isnumeric(): # probably the seed
            # no need to remove quotation marks, just comma
            return int(s[:-1]) # only used as a label, so ok to leave as str

        return s[1:-2]

    f = open(filename,'r')

    # list of lists, one for each param
    param_lst_all = []
    proposal_lst_all = []
    # list of list, store [truth, min, max] for each param
    param_search_space_lst = []

    # evaluation metrics
    #heuristic size
    #dis_loss,gen_loss,real_acc,fake_acc for rows, iter for columns
    #iter = 300
    eval_metrics = np.full((4,300),np.nan)


    num_param = None
    param_names = None
    training = False

    trial_data = {}
    line_no = 0
    for line in f:
        line_no += 1
        if line.startswith("ITER"):
            training = True
            iter = int(line.split(" ")[-1])
            #initiate dis_loss incase first iteration is not accepted

        if line.startswith("{"):
            tokens = line.split()
            print(tokens)
            param_str = tokens[3][1:-2]
            print("PARAMS", param_str)
            param_names = param_str.split(",")
            num_param = len(param_names)
            for i in range(num_param):
                param_lst_all.append([])
                proposal_lst_all.append([])

            trial_data['model'] = clean_param_tkn(tokens[1])
            trial_data['params'] = param_str
            trial_data['data_h5'] = clean_param_tkn(tokens[5])
            trial_data['bed_file'] = clean_param_tkn(tokens[7])
            trial_data['reco_folder'] = clean_param_tkn(tokens[9])
            trial_data['seed'] = clean_param_tkn(tokens[15])
            trial_data['sample_sizes'] = clean_param_tkn(tokens[17])
        
        elif param_names != None and line.startswith(tuple(param_names)):
            Name, TRUTH, MIN, MAX = line.strip().split("\t")
            param_search_space_lst.append((Name, float(TRUTH), float(MIN), float(MAX)))
        
        elif training and "Epoch" in line:
            tokens = line.split()
            disc_loss = float(tokens[3][:-1])
            real_acc = float(tokens[6][:-1])/100
            fake_acc = float(tokens[9])/100

            eval_metrics[0, iter] = disc_loss
            eval_metrics[2, iter] = real_acc
            eval_metrics[3, iter] = fake_acc

        elif "T, p_accept" in line:
            tokens = line.split()
            # parse current params and add to each list
            mini_lst = parse_mini_lst(tokens[-1-num_param:-1])
            #add generater loss to eval_metrics
            eval_metrics[1, iter] = float(tokens[-1])
            add_to_lst(param_lst_all, mini_lst)

        elif "proposal" in line:
            tokens = line.split()
            # parse current params and add to each list
            mini_lst = parse_mini_lst(tokens[2:-1])
            add_to_lst(proposal_lst_all, mini_lst)
    f.close()

    # Use -1 instead of iter for the last iteration
    final_params = [param_lst_all[i][-1] for i in range(num_param)]
    
    final_discriminator_acc = (real_acc + fake_acc) / 2
    if return_acc:
        return final_params, eval_metrics, \
            trial_data, param_search_space_lst, param_lst_all, proposal_lst_all, final_discriminator_acc
    else:
        return final_params, trial_data
    
def plot_parse_output(eval_metrics, param_search_space_lst, param_lst_all, proposal_lst_all, final_discriminator_acc, \
                        output_filepath, output_filepath_2, output_filepath_3):
    
    eval_metrics = forward_fill(eval_metrics)

    disc_loss_lst, gen_loss_lst, real_acc_lst, fake_acc_lst = eval_metrics[0, :], eval_metrics[1, :], eval_metrics[2, :], eval_metrics[3, :]
    fig, ax = plt.subplots(2, 1, figsize=(20, 30))
    ax[0].plot(range(len(gen_loss_lst)), gen_loss_lst, 'r', label = "generator loss")
    ax[0].plot(range(len(disc_loss_lst)), disc_loss_lst, 'b', label = "discriminator loss")
    ax[1].axhline(y=0.5, color='g', linestyle='--')
    ax[1].plot(range(len(real_acc_lst)), real_acc_lst, 'r', label = "discriminator real_acc")
    ax[1].plot(range(len(fake_acc_lst)), fake_acc_lst, 'b', label = "discriminator fake_acc")
    ax[1].plot(range(len(real_acc_lst)), [(x+y)/2 for x,y in zip(*[real_acc_lst, fake_acc_lst])], 'g', label = "discriminator avg_acc")
    ax[1].title.set_text("ITER {}  disc_avg_acc = {:.2f}".format(len(real_acc_lst), final_discriminator_acc))
    

    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(output_filepath, bbox_inches='tight')

    fig, ax = plt.subplots(len(param_lst_all), 1, figsize=(20, 30))
    for index, (param, param_search_space) in enumerate(zip(param_lst_all, param_search_space_lst)):
        ax[index].plot(range(len(param)), param, label = param_search_space[0])
        ax[index].axhline(y=param_search_space[1], color='g', linestyle='-', linewidth='0.2')
        ax[index].set(xlabel="iters", ylabel=param_search_space[0])
        ax[index].axhline(y=param_search_space[2], color='r', linestyle='-')
        ax[index].axhline(y=param_search_space[3], color='r', linestyle='-')
        ax[index].set_ylim([param_search_space[2] * 0.7, param_search_space[3] * 1.2])
        ax[index].legend(["ITER {}  {} = {:.1f}".format(len(param),param_search_space[0], param[-1]), "Baseline {} = {}".format(param_search_space[0], param_search_space[-3])])

    plt.savefig(output_filepath_2, dpi=350)

    fig, ax = plt.subplots(len(proposal_lst_all), 1, figsize=(20, 30))
    for index, (param, param_search_space) in enumerate(zip(proposal_lst_all, param_search_space_lst)):
        ax[index].plot(range(len(param)), param, label = param_search_space[0], linewidth='0.5')
        ax[index].set(xlabel="proposals", ylabel=param_search_space[0])
        ax[index].axhline(y=param_search_space[1], color='g', linestyle='-')
        ax[index].axhline(y=param_search_space[2], color='r', linestyle='-')
        ax[index].axhline(y=param_search_space[3], color='r', linestyle='-')
        ax[index].set_ylim([param_search_space[2] * 0.7, param_search_space[3] * 1.2])

    plt.savefig(output_filepath_3, dpi=350)

def forward_fill(arr):
    '''
        forward-fill NaN values in numpy array
    '''
    mask = np.isnan(arr)
    idx = np.where(~mask,np.arange(mask.shape[1]),0)
    np.maximum.accumulate(idx,axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:,None], idx]
    return out

def find_most_confused_state(eval_metrics):
    min_acc = 5000
    min_acc_iter = None
    avg_acc_reg_lst = []
    real_acc_lst, fake_acc_lst = eval_metrics[2, :], eval_metrics[3, :]
    num_iter = len(real_acc_lst)
    for iter, (real_acc, fake_acc) in enumerate(zip(real_acc_lst, fake_acc_lst)):
        if real_acc and fake_acc == np.nan:
            continue
        avg_acc = abs(real_acc - 0.5) + abs(fake_acc - 0.5)
        T = 1 - iter / num_iter
        avg_acc = avg_acc / T
        if min_acc > avg_acc:
            min_acc = avg_acc
            min_acc_iter = iter
    return min_acc_iter

        
        
    
        









################################################################################
# COMPUTE STATS
################################################################################

def compute_sfs(vm):
    """Show the beginning of the SFS"""
    ac = vm.count_alleles()
    return [variant_counts(ac,i) for i in range(0,NUM_SFS)]

def variant_counts(ac,c):
    """Helper for SFS"""
    count = 0
    for site in np.array(ac):
        # this should not happen if VCF is pre-processed correctly
        if len(site) > 2:
            print("non-bi-allelic?", site)

        # non-seg can happen after splitting into separate pops
        elif len(site) == 1:
            if c == 0:
                count += 1

        # folded but it shouldn't matter b/c we do major=0 and minor=1
        elif site[0] == c or site[1] == c:
            count += 1

    return count

def compute_ld(vm, L):
    """Compute LD as a function of inter-SNP distance"""

    stringy_data = [''.join(map(str, row)) for row in vm.data.transpose()]
    sd = libsequence.SimData(vm.positions, stringy_data)
    ld = libsequence.ld(sd)

    # num bins
    nbin = NUM_LD
    max_dist = 5000 # TODO make flexible! (5k for mosquito, 20k for human)
    dist_bins = np.linspace(0,max_dist,nbin)
    rsquared = [0]*nbin
    counts = [0]*nbin

    # compute bin distances and add ld values to the distance class
    for dict in ld:
        snp_dist = abs(dict['i'] - dict['j'])*L # L is region length
        rsq = dict['rsq']
        idx = np.digitize(snp_dist, dist_bins, right=False)

        # after last num is okay, just goes in last bin
        if idx < 0:
            print(idx, "LD problem!")
        rsquared[idx-1] += rsq
        counts[idx-1] += 1

    # average rsq
    for i in range(nbin):
        if counts[i] > 0:
            rsquared[i] = rsquared[i]/counts[i]
    return rsquared

def compute_stats(vm, vm_region):
    """Generic stats for vm (fixed num SNPs) and vm_region (fixed region len)"""

    stats = []
    ac = vm.count_alleles()
    ac_region = vm_region.count_alleles()

    # Tajima's D (use region here - not fixed num SNPs)
    stats.append(libsequence.tajd(ac_region))

    # pi
    #stats.append(libsequence.thetapi(ac))

    # wattersons
    stats.append(libsequence.thetaw(ac))

    # num haps
    stats.append(libsequence.number_of_haplotypes(vm))

    return stats

def compute_fst(raw):
    """
    FST (for two populations)
    https://scikit-allel.readthedocs.io/en/stable/stats/fst.html
    """
    # raw has been transposed
    nvar = raw.shape[0]
    nsam = raw.shape[1]
    raw = np.expand_dims(raw, axis=2).astype('i')

    g = allel.GenotypeArray(raw)
    subpops = [range(nsam//2), range(nsam//2, nsam)]

    # for each pop
    ac1 = g.count_alleles(subpop=subpops[0])
    ac2 = g.count_alleles(subpop=subpops[1])

    # compute average fst
    num, den = allel.hudson_fst(ac1, ac2)
    fst = np.sum(num) / np.sum(den)
    return fst

################################################################################
# PLOTTING FUNCTIONS
################################################################################

def plot_generic(ax, name, real, sim, real_color, sim_color, pop="",
    sim_label="", single=False):
    """Plot a generic statistic."""
    # SFS
    if name == "minor allele count (SFS)":
        # average over regions
        num_sfs = len(real)
        real_sfs = [sum(rs)/num_sfs for rs in real]
        sim_sfs = [sum(ss)/num_sfs for ss in sim]
                
        
        ax.bar([x -0.3 for x in range(num_sfs)], real_sfs, label=pop, width=0.4,
            color=real_color)
        ax.bar(range(num_sfs), sim_sfs, label=sim_label, width=0.4,
            color=sim_color)
        ax.set_xlim(-1,len(real_sfs))
        ax.set_ylabel("frequency per region")
        #ax.text(0, 0, diff)

    # LD
    elif name == "distance between SNPs":
        nbin = NUM_LD
        max_dist = 500 # TODO make flexible! (5k for mosquito, 20k for human)
        dist_bins = np.linspace(0,max_dist,nbin)
        real_mean = [np.mean(rs) for rs in real]
        sim_mean = [np.mean(ss) for ss in sim]
        real_stddev = [np.std(rs) for rs in real]
        sim_stddev = [np.std(ss) for ss in sim]

        # plotting
        ax.errorbar(dist_bins, real_mean, yerr=real_stddev, color=real_color,
            label=pop)
        ax.errorbar([x for x in dist_bins], sim_mean, yerr=sim_stddev,
            color=sim_color, label=sim_label)
        ax.set_ylabel(r'LD ($r^2$)')

    # all other stats
    else:
        sns.histplot(real, ax=ax, color=real_color, label=pop, kde=True,
            stat="density", edgecolor=None)
        sns.histplot(sim, ax=ax, color=sim_color, label=sim_label, kde=True,
            stat="density", edgecolor=None)

    # inter-SNP distances
    if name == "inter-SNP distances":
        ax.set_xlim(-25,100)
    ax.set(xlabel=name)

    # legend
    if single or name == "Hudson's Fst":
        ax.legend()
    else:
        if len(pop) > 3:
            x_spacing = 0.83
        else:
            x_spacing = 0.85

        ax.text(x_spacing, 0.85, pop, horizontalalignment='center',
            transform=ax.transAxes, fontsize=18)
        
def plot_generic_with_baseline(ax, name, real, sim, baseline, real_color, sim_color, baseline_color, pop="",
    sim_label="", baseline_label="", single=False):
    """Plot a generic statistic."""
    
    round_val = 6

    # SFS
    if name == "minor allele count (SFS)":
        # average over regions
        num_sfs = len(real)
        real_sfs = [sum(rs)/num_sfs for rs in real]
        sim_sfs = [sum(ss)/num_sfs for ss in sim]
        baseline_sfs = [sum(bs)/num_sfs for bs in baseline]
        
        sim_diff = calc_distribution_dist(real_sfs, sim_sfs)
        baseline_diff = calc_distribution_dist(real_sfs, baseline_sfs)
        text = "sim_wass_dist:" + str(round(sim_diff, round_val)) + "\n" + "baseline_wass_dist:" + str(round(baseline_diff, round_val))

        
        
        ax.bar([x -0.3 for x in range(num_sfs)], real_sfs, label=pop, width=0.3,
            color=real_color)
        ax.bar(range(num_sfs), sim_sfs, label=sim_label, width=0.3,
            color=sim_color)
        ax.bar([x +0.3 for x in range(num_sfs)], baseline_sfs, label=baseline_label, width=0.3,
            color=baseline_color)
        ax.set_xlim(-1,len(real_sfs))
        ax.set_ylabel("frequency per region")
        ax.text(.01, .99, text, fontsize=8, ha='left', va='top', transform=ax.transAxes)

    # LD
    elif name == "distance between SNPs":
        nbin = NUM_LD
        max_dist = 500 # TODO make flexible! (5k for mosquito, 20k for human)
        dist_bins = np.linspace(0,max_dist,nbin)
        real_mean = [np.mean(rs) for rs in real]
        sim_mean = [np.mean(ss) for ss in sim]
        real_stddev = [np.std(rs) for rs in real]
        sim_stddev = [np.std(ss) for ss in sim]
        baseline_mean = [np.mean(bs) for bs in baseline]
        baseline_stddev = [np.std(bs) for bs in baseline]

        sim_diff = calc_distribution_dist(real_mean, sim_mean)
        baseline_diff = calc_distribution_dist(real_mean, baseline_mean)
        text = "sim_wass_dist:" + str(round(sim_diff, round_val)) + "\n" + "baseline_wass_dist:" + str(round(baseline_diff, round_val))
      
        
        # plotting
        ax.errorbar(dist_bins, real_mean, yerr=real_stddev, color=real_color,
            label=pop)
        ax.errorbar([x for x in dist_bins], sim_mean, yerr=sim_stddev,
            color=sim_color, label=sim_label)
        ax.errorbar([x for x in dist_bins], baseline_mean, yerr=baseline_stddev,
            color=baseline_color, label=baseline_label)
        ax.set_ylabel(r'LD ($r^2$)')
        ax.text(.01, .99, text, fontsize=8, ha='left', va='top', transform=ax.transAxes)

    # all other stats
    else:
        sns.kdeplot(real, ax=ax, color=real_color, label=pop,
            common_norm=False)
        sns.kdeplot(sim, ax=ax, color=sim_color, label=sim_label, 
            common_norm=False)
        sns.kdeplot(baseline, ax=ax, color=baseline_color, label=baseline_label, 
            common_norm=False)
        sim_diff = calc_distribution_dist(real, sim)
        baseline_diff = calc_distribution_dist(real, baseline)
        text = "sim_wass_dist:" + str(round(sim_diff, round_val)) + "\n" + "baseline_wass_dist:" + str(round(baseline_diff, round_val))
        ax.text(.01, .99, text, fontsize=8, ha='left', va='top', transform=ax.transAxes)


    # inter-SNP distances
    if name == "inter-SNP distances":
        ax.set_xlim(-50,100)
    ax.set(xlabel=name)

    # legend
    if single or name == "Hudson's Fst":
        ax.legend()
    else:
        if len(pop) > 3:
            x_spacing = 0.83
        else:
            x_spacing = 0.85

        ax.text(x_spacing, 0.85, pop, horizontalalignment='center',
            transform=ax.transAxes, fontsize=18)

################################################################################
# COLLECT STATISTICS
################################################################################

def stats_all(matrices, matrices_region, L = global_vars.L):
    """Set up and compute stats"""

    # sfs
    pop_sfs = []
    for j in range(NUM_SFS):
        pop_sfs.append([])
    # inter-snp
    pop_dist = []
    # LD
    pop_ld = []
    for j in range(NUM_LD):
        pop_ld.append([])
    # Taj D, pi, num haps
    pop_stats = []
    for j in range(3):
        pop_stats.append([])

    # go through each region
    for i in range(len(matrices)):

        # fixed SNPs
        matrix = matrices[i]
        raw = matrix[:,:,0].transpose()
        intersnp = matrix[:,:,1][0] # all the same
        pos = [sum(intersnp[:i]) for i in range(len(intersnp))]
        assert len(pos) == len(intersnp)
        print("num SNPs", len(pos))
        vm = libsequence.VariantMatrix(raw, pos)

        # fixed region
        matrix_region = matrices_region[i]
        raw_region = matrix_region[:,:,0].transpose()
        intersnp_region = matrix_region[:,:,1][0] # all the same
        pos_region = [sum(intersnp_region[:i]) for i in
            range(len(intersnp_region))]
        assert len(pos_region) == len(intersnp_region)
        print("num SNPs region", len(pos_region))
        vm_region = libsequence.VariantMatrix(raw_region, pos_region)

        # sfs
        sfs = compute_sfs(vm)
        for s in range(len(sfs)):
            pop_sfs[s].append(sfs[s])

        # inter-snp
        pop_dist.extend([x*L for x in intersnp])

        # LD
        ld = compute_ld(vm, L)
        for l in range(len(ld)):
            pop_ld[l].append(ld[l])

        # rest of stats
        stats = compute_stats(vm, vm_region)
        for s in range(len(stats)):
            pop_stats[s].append(stats[s])

        input('enter')
    return [pop_sfs, pop_dist, pop_ld] + pop_stats

def fst_all(matrices):
    """Fst for all regions"""
    real_fst = []
    for i in range(len(matrices)):
        matrix = matrices[i]

        raw = matrix[:,:,0].transpose()
        intersnp = matrix[:,:,1][0] # all the same

        fst = compute_fst(raw)
        real_fst.append(fst)

    return real_fst

def calc_distribution_dist(dist_p, dist_q):
    """Calculate wasserstein distance distribution P and distribution Q

    Args:
        dist_p (list of float): distribution P
        dist_q (list of float): distribution P

    Returns:
        float: wasserstein distance between distribution P and distribution Q
    """
    return wasserstein_distance(dist_p, dist_q)

    
    

    
