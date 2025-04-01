"""
Script to filter non-segregating sites from an h5. Produces another h5 file.
Author: Sara Mathieson
Date: 3/31/25
"""

import h5py
import optparse
import numpy as np
import sys

# example command line
# python3 filter_nonseg.py -i YRI.h5 -o YRI_filter.h5

def main():
    opts = parse_args()
    filter(opts.in_filename, opts.out_filename)

def parse_args():
    """Parse command line arguments."""
    parser = optparse.OptionParser(description='Remove non-seg sites from h5')

    parser.add_option('-i', '--in_filename', type='string', \
        help='path to input h5 file')
    parser.add_option('-o', '--out_filename', type='string', \
        help='path to output h5 file')

    (opts, args) = parser.parse_args()

    mandatories = ['in_filename','out_filename']
    for m in mandatories:
        if not opts.__dict__[m]:
            print('mandatory option ' + m + ' is missing\n')
            parser.print_help()
            sys.exit()

    return opts

def filter(in_filename, out_filename):
    """Convert vcf_filename"""
    # here we save only CHROM, GT (genotypes) and POS (SNP positions)
    # original fields should be: ['CHROM','GT','POS']

    callset = h5py.File(in_filename, mode='r')
    print(list(callset.keys()))
    # output: ['GT'] ['CHROM', 'POS']
    print(list(callset['calldata'].keys()),list(callset['variants'].keys()))

    raw = callset['calldata/GT']
    print("raw", raw.shape)
    newshape = (raw.shape[0], -1)
    haps_all = np.reshape(raw, newshape)
    pos_all = callset['variants/POS']
    # same length as pos_all, noting chrom for each variant (sorted)
    chrom_all = callset['variants/CHROM']
    print("after haps", haps_all.shape)
    num_snps, num_samples = haps_all.shape
    print("num snps", num_snps, "num samples", num_samples)

    '''print(self.pos_all.shape)
    print(self.pos_all.chunks)
    print(self.chrom_all.shape)
    print(self.chrom_all.chunks)'''
    assert num_snps == len(pos_all) # total for all chroms

    new_chroms = []
    new_haps = []
    new_pos = []
    for s in range(20): # num_snps): TODO put back, just for testing
        # get row, assert len is num_haps
        row = haps_all[s]
        assert len(row) == num_samples

        # check if non-seg (all 0's or all 1's?)
        unique, counts = np.unique(row, return_counts=True)
        if len(unique) > 2: # multi-allelic
            print("multi-allelic!", unique, counts)
        elif len(unique) == 2: # if segregating, add to list for CHROM, GT, POS
            new_chroms.append(chrom_all[s])
            new_haps.append(row)
            new_pos.append(pos_all[s])

    callset.close()
    print("frac kept", len(new_pos)/len(pos_all))

    # save h5

    with h5py.File(out_filename, 'w') as new_file:

        group1 = new_file.create_group('calldata')
        group2 = new_file.create_group('variants')

        new_file['calldata/GT'] = new_haps
        new_file['variants/POS'] = new_pos
        new_file['variants/CHROM'] = new_chroms
        
        #dset11 = group1.create_dataset('my_dataset11', data=dat, compression=9)
        #dset12 = group1.create_dataset('my_dataset12', data=dat, compression=9)
        #dset13 = group1.create_dataset('my_dataset13', data=dat, compression=9)

        '''groups=list(f.keys())

        grp=f[groups[0]]
        dataset=list(grp.keys())

        for each in dataset:
            grp[each].attrs['env']='cloudy'
            grp[each].attrs['temp']=25
            #grp[each]._f_setattr('cloudy', 'True')

        grp=f[groups[1]]
        dataset=list(grp.keys())

        for each in dataset:
            grp[each].attrs['env']='rainy'
            grp[each].attrs['temp']=20
            #grp[each]._f_setattr('rainy', 'True')

        for each_grp in groups:
            dataset=list(f[each_grp].keys())
            for each_ds in dataset:
                print ('For ', each_grp, '.', each_ds,':')
                print ('\tenv =', f[each_grp][each_ds].attrs['env'])
                print ('\ttemp=',f[each_grp][each_ds].attrs['temp'])'''

    new_file.close()

    # check new file, how to put back into haps?

    callset = h5py.File(out_filename, mode='r')
    print(list(callset.keys()))
    # output: ['GT'] ['CHROM', 'POS']
    print(list(callset['calldata'].keys()),list(callset['variants'].keys()))

    raw = callset['calldata/GT']
    print("raw", raw.shape)
    #newshape = (raw.shape[0], -1)
    #haps_all = np.reshape(raw, newshape)
    pos_all = callset['variants/POS']
    print(pos_all)
    # same length as pos_all, noting chrom for each variant (sorted)
    chrom_all = callset['variants/CHROM']
    print(chrom_all)
    #print("after haps", haps_all.shape)
    #num_snps, num_samples = haps_all.shape
    #print("num snps", num_snps, "num samples", num_samples)

if __name__ == "__main__":
    main()
