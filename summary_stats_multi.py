"""
Compare summary statistics from real data with data simulated under the
inferred parameters, as well as baseline parameter from same demographic model
Author: Sara Mathieson, Rebecca Riley, Jacky Siu Pui Chung
Date: 06/09/23
"""

# python imports
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

# our imports
import global_vars
import ss_helpers
import util

# globals
NUM_TRIAL = 10 #200
# statistic names
NAMES = [
    "minor allele count (SFS)",
    "inter-SNP distances",
    "distance between SNPs",
    #"Tajima's D",
    r'pairwise heterozygosity ($\pi$)', "Watterson",
    "number of haplotypes",
    "Hudson's Fst"]
FST_COLOR = "purple"

# for ooa2 (YRI/CEU) (no longer supported)
#FSC_PARAMS = [21017, 0.0341901, 3105.5, 21954, 33077.5, 2844, 1042]

DADI = True

# for dadi (mosquito, BFS population)
#DADI_NAMES = ["Na", "N1", "N2", "T1", "T2"]
#DADI_PARAMS = [384845.04236326, 1891371.2275129908, 11140821.633397933, 60447.09280337712, 22708.95299729848]

#DADI_NAMES = ["NI", "TG", "NF", "TS", "NI1", "NI2", "NF1", "NF2"]

# baseline parameters for GNB-BFA_gamb_nsg or GN-BF_gam_biallelic_2017
print("GN-BF")
DADI_PARAMS = [420646, 89506, 9440437, 2245, 18328570, 42062652, 42064645, 42064198]
# SM: changed migration param from 60 to 20 to reflect 2017 paper
DADI_MIG_PARAMS = [415254, 93341, 8292759, 11637, 2635696, 2748423, 11101754, 11439976, 20]

# baseline parameters for CM-UG_gam_nsg
#print("CM-UG")
#DADI_PARAMS = [432139, 84723, 11040070, 3377, 38787744, 24729852, 13116347, 3575499]
#DADI_MIG_PARAMS = [282184, 134544, 9153012, 34117, 169035, 28194588, 10251626, 2938915, 4.363307495]


def main():
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    demo = sys.argv[3]

    print("input file", input_file)
    print("output file", output_file)
    print("baseline demographic model", demo)

    if global_vars.OVERWRITE_TRIAL_DATA:
        in_file_data = global_vars.TRIAL_DATA
        value_strs = global_vars.TRIAL_DATA['param_values'].split(',')
        param_values = [float(value_str) for value_str in value_strs]
        assert len(param_values) == len(in_file_data['params'].split(','))
    else:
        param_values, in_file_data = ss_helpers.parse_output(input_file, return_acc=False)

    

    opts, param_values = util.parse_args(in_file_data = in_file_data,
        param_values=param_values)
    generator, iterator, parameters, sample_sizes = util.process_opts(opts,
        summary_stats=True)

    #title_data = get_title_from_trial_data(opts, param_values,
    #    generator.sample_sizes) if global_vars.SS_SHOW_TITLE else None
    
    pop_names = opts.data_h5.split("/")[-1].split(".")[0] \
                       if opts.data_h5 is not None else ""
    
    print("pop_names", pop_names)
    # sets global_vars.SS_LABELS and global_vars.SS_COLORS
    # overwrite this function in globals.py to change
    global_vars.update_ss_labels(pop_names, num_pops=len(generator.sample_sizes))
    #param_values = [593472.1, 20096.4, 64951692.5, 1832.7, 279568717, 21202531, 79263319, 2056255290]
    #param_values = [583783.8528353142, 20096.41648961899, 66115552.33895082, 2167.5516132997705, 84610222.15209761, 16981864.649424005, 282202142.90886563, 213952213.9980046]
    generator.update_params(param_values)


    print("VALUES", param_values)
    print("made it through params")

    # use the parameters we inferred!
    # fsc=False
    # if opts.model == 'fsc':
    #     print("\nALERT you are running FSC sim!\n")
    #     print("FSC PARAMS!", FSC_PARAMS)
    #     generator.update_params(FSC_PARAMS) # make sure to check the order!
    #     fsc=True
    # elif DADI:
    #     print("\nALERT you are running DADI!\n")
    #     print("DADI PARAMS!", DADI_NAMES, DADI_PARAMS)
    #     generator.param_names = DADI_NAMES
    #     generator.update_params(DADI_PARAMS) # make sure to check the order!

    '''
    NOTE: for summary stats, use neg1=False to keep hap data as 0/1 (not -1/1)
    NOTE: use region_len=True for Tajima's D (i.e. not all regions have same S)
    '''
    # real
    real_matrices = iterator.real_batch(batch_size=NUM_TRIAL, neg1=False)
    print("finish real")
    real_matrices_region = iterator.real_batch(batch_size=NUM_TRIAL, neg1=False,
        region_len=True)
    print("finish real region_len")
    # sim
    sim_matrices = generator.simulate_batch(batch_size=NUM_TRIAL, neg1=False)
    print("finish sim")
    sim_matrices_region = generator.simulate_batch(batch_size=NUM_TRIAL,
        neg1=False, region_len=True)
    print("finish sim region_len")
    
    #TODO: code to be replaced to load different demographic parameters baseline from the input outfile, 
    # and create new generator to compare different demographic model
    if demo == "dadi_joint":  
        generator.update_params(DADI_PARAMS)
    elif demo == "dadi_joint_mig":
        generator.update_params(DADI_MIG_PARAMS)
    else:
        print("unknown demographic model baseline")
    sim_baseline_matrices = generator.simulate_batch(batch_size=NUM_TRIAL, neg1=False)
    print("finish sim baseline")
    sim_baseline_matrices_region = generator.simulate_batch(batch_size=NUM_TRIAL,
        neg1=False, region_len=True)
    print("finish sim baseline region_len")
    
    

    '''# go through each region
    for i in range(len(sim_matrices)):

        # fixed SNPs
        matrix = sim_matrices[i]
        raw = matrix[:,:,0].transpose()

        # check neg1
        unique, counts = np.unique(raw, return_counts=True)
        if len(unique) > 2:
            print("hap_data", dict(zip(unique, counts)))
            input('enter')

    input('pause')'''

    num_pop = len(sample_sizes)

    # one pop models
    if num_pop == 1:
        nrows, ncols = 3, 2
        size = (7, 7)
        first_pop, second_pop = [], []

    # two pop models
    elif num_pop == 2:
        nrows, ncols = 4, 4
        size = (14, 10)
        first_pop, second_pop = [0], [1]

    # OOA3
    elif opts.model in ['ooa3']:
        nrows, ncols = 6, 4
        size = (14, 14)
        first_pop, second_pop = [0, 0, 1], [1, 2, 2]

    else:
        print("unsupported number of pops", num_pop)

    # split into individual pops
    
    real_all, real_region_all = split_matrices(real_matrices, real_matrices_region, sample_sizes)
    sim_all, sim_region_all = split_matrices(sim_matrices, sim_matrices_region, sample_sizes)
    sim_baseline_all, sim_baseline_region_all = split_matrices(sim_baseline_matrices, sim_baseline_matrices_region, sample_sizes)


    # stats for all populations
    real_stats_lst = []
    sim_stats_lst = []
    sim_baseline_stats_lst = []
    for p in range(num_pop):
        print("real stats for pop", p)
        #temp solution for fixing difference in coalescent tree length
        # if "sim" in input_file:
        #     print("reading sim file")
        #     real_stats_pop = ss_helpers.stats_all(real_all[p], real_region_all[p], 5000000)
        # else:
        real_stats_pop = ss_helpers.stats_all(real_all[p], real_region_all[p])
        print("sim stats for pop", p)
        sim_stats_pop = ss_helpers.stats_all(sim_all[p], sim_region_all[p])
        print("sim baseline stats for pop", p)
        sim_baseline_stats_pop = ss_helpers.stats_all(sim_baseline_all[p], sim_baseline_region_all[p])

        real_stats_lst.append(real_stats_pop)
        sim_stats_lst.append(sim_stats_pop)
        sim_baseline_stats_lst.append(sim_baseline_stats_pop)

    print("got through main stats")

    # Fst over all pairs
    real_fst_lst = []
    sim_fst_lst = []
    sim_baseline_fst_lst = []
    for pi in range(len(first_pop)):
        a = first_pop[pi]
        b = second_pop[pi]
        real_ab = np.concatenate((np.array(real_all[a]), np.array(real_all[b])),
            axis=1)
        sim_ab = np.concatenate((np.array(sim_all[a]), np.array(sim_all[b])),
            axis=1)
        sim_baseline_ab = np.concatenate((np.array(sim_baseline_all[a]), np.array(sim_baseline_all[b])),
            axis=1)

        # compute Fst
        real_fst = ss_helpers.fst_all(real_ab, sample_sizes)
        sim_fst = ss_helpers.fst_all(sim_ab, sample_sizes)
        sim_baseline_fst = ss_helpers.fst_all(sim_baseline_ab, sample_sizes)
        real_fst_lst.append(real_fst)
        sim_fst_lst.append(sim_fst)
        sim_baseline_fst_lst.append(sim_baseline_fst)

    print("got through fst")
    
    # finall plotting call
    plot_stats_all(nrows, ncols, size, real_stats_lst, sim_stats_lst,sim_baseline_stats_lst,
        real_fst_lst, sim_fst_lst, sim_baseline_fst_lst, output_file)

def split_matrices(matrices, matrices_region, sample_sizes):

    # set up empty arrays
    _all, region_all = [], []

    start_idx = 0
    for s in sample_sizes:
        end_idx = start_idx + s

        # parse matrices
        p = matrices[:,start_idx:end_idx,:,:]
        region_p = []
        for item in matrices_region:
            region_p.append(item[start_idx:end_idx,:,:])
        _all.append(p)
        region_all.append(region_p)

        # last step: update start_idx
        start_idx = end_idx

    return _all, region_all

# one, two, and three pops
def plot_stats_all(nrows, ncols, size, real_stats_lst, sim_stats_lst, sim_baseline_stats_lst,
    real_fst_lst, sim_fst_lst, sim_baseline_fst_lst, output, title_data=None):
    num_pop = len(real_stats_lst)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=size)

    if title_data is not None:
        fig.suptitle(title_data["title"], fontsize=title_data["size"])
    
    # labels and colors
    labels = global_vars.SS_LABELS[:num_pop]
    sim_label = global_vars.SS_LABELS[-1]
    #TEMP
    sim_baseline_label = "dadi"
    colors = global_vars.SS_COLORS[:num_pop]
    sim_color = global_vars.SS_COLORS[-1]
    #TEMP
    sim_baseline_color = "green"

    # plot each population
    rows = [0, 0, 3]
    cols = [0, 2, 2]
    single = True if num_pop == 1 else False
    for p in range(num_pop): # one/two pop won't use last indices
        real_color = colors[p]
        real_label = labels[p]
        real_pop = real_stats_lst[p]
        sim_pop = sim_stats_lst[p]
        sim_baseline_pop = sim_baseline_stats_lst[p]
        plot_population(axes, rows[p], cols[p], real_color, real_label,
            real_pop, sim_color, sim_label, sim_pop, 
            sim_baseline_color, sim_baseline_label, sim_baseline_pop, single=single)

    # Fst (all pairs)
    cidx = 0
    first_pop = [0, 0, 1]
    second_pop = [1, 2, 2]
    if num_pop == 2:
        cidx = 1
    for pi in range(len(real_fst_lst)): # pi -> pair index
        pair_label = labels[first_pop[pi]] + "/" + labels[second_pop[pi]]
        #load custom code that intake three sets of summary statistics for plotting
        ss_helpers.plot_generic_with_baseline(axes[3+pi][cidx], NAMES[6], real_fst_lst[pi],
            sim_fst_lst[pi], sim_baseline_fst_lst[pi], FST_COLOR, sim_color, sim_baseline_color, pop=pair_label,
            sim_label=sim_label, baseline_label=sim_baseline_label)

    # overall legend
    if num_pop >= 2:
        for p in range(num_pop):
            p_real = mpatches.Patch(color=colors[p], label=labels[p] + \
                ' real data')
            p_sim = mpatches.Patch(color=sim_color, label=labels[p] + \
                ' sim data')
            p_sim_baseline = mpatches.Patch(color=sim_baseline_color, label=sim_baseline_label + \
                ' baseline')
            if num_pop == 2:
                axes[3][0+3*p].axis('off')
                axes[3][0+3*p].legend(handles=[p_real, p_sim, p_sim_baseline], loc=10,
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
    sim_label, sim_tuple, sim_baseline_color, sim_baseline_label, sim_baseline_tuple, single=False):
    """
    Plot all 6 stats for a single population, starting from the (i,j) subplot.
    """
    for r in range(3):
        for c in range(2):
            idx = 2*r+c
            ss_helpers.plot_generic_with_baseline(axes[i+r][j+c], NAMES[idx], real_tuple[idx],
                sim_tuple[idx], sim_baseline_tuple[idx], real_color, sim_color, sim_baseline_color, pop=real_label,
                sim_label=sim_label, baseline_label=sim_baseline_label,single=single)

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
