import h5py
import sys
import numpy as np
import allel
import argparse


def subset_hdf5(h5py_file, input_pass, output_pass, input_populations):
    callset = h5py.File(h5py_file, mode='r')
    chrom = h5py_file.split('.')[4]
    print('CHROM', chrom)
    samples = callset[chrom]['samples'][:]
    samples = [sample.decode("utf-8") for sample in samples]
    print('samples length:', len(samples))
    sample_populations = []
    with open(input_populations, 'r') as file:
        for line in file:
            sample_populations.append(line.strip())
    loc_samples = np.isin(np.array(samples), np.array(sample_populations))
    true_samples = [sample for sample, is_true in zip(samples, loc_samples) if is_true]
    print('filtered samples length', len(true_samples))

    variants = allel.VariantChunkedTable(callset[chrom]['variants'])
    pos = variants['POS'][:]

    pass_filters = []
    with open(input_pass, 'r') as file:
        for line in file:
            pass_filters.append(int(line.strip()))

    # Remove positions that lie within Tep1
    if chrom == '3L':
        start = 11201863
        end = 11206930
    elif chrom == '3R':
        start = 32058284
        end = 32063172
    pass_filters_set = set(pass_filters)
    pass_filters_set = [x for x in pass_filters if not (start <= x <= end)]
    print('pass_filters_set length:', len(pass_filters_set))

    # Convert lists to NumPy arrays for vectorized operations
    np_pos = np.array(pos)
    np_pass_filters_set = np.array(list(pass_filters_set))

    loc_variant_selection_numpy = np.isin(np_pos, np_pass_filters_set)
    loc_variant_selection = loc_variant_selection_numpy.tolist()
    print('loc_variant_selection length:', len(loc_variant_selection))
    print('number of variants removed', loc_variant_selection.count(False))
    true_positions = [pos for pos, is_true in zip(pos, loc_variant_selection) if is_true]
    print('number of remained variants:', len(true_positions))

    assert len(loc_variant_selection) == len(true_positions) + loc_variant_selection.count(False)

    gt = callset['/{}/calldata/genotype'.format(chrom)]
    gt_dask = allel.GenotypeDaskArray(gt)
    print(gt_dask.shape)
    gt_subset = gt_dask.subset(loc_variant_selection, loc_samples).compute()
    print(gt_subset.shape)

    assert len(true_positions) == gt_subset.shape[0], (len(true_positions), gt_subset.shape[0])
    assert len(true_samples) == gt_subset.shape[1], (len(true_samples), gt_subset.shape[1])

    with h5py.File(output_pass, 'w') as file:
        # Creating 'variants' group with gzip compression
        variants_group = file.create_group('variants')
        variants_group.create_dataset('CHROM', data=[chrom] * gt_subset.shape[0], compression='gzip')
        variants_group.create_dataset('POS', data=true_positions, compression='gzip')

        # Creating 'calldata' group with gzip compression
        calldata_group = file.create_group('calldata')
        calldata_group.create_dataset('GT', data=gt_subset, compression='gzip')
        
    return


def concatenate_hdf5_files(file1_path, file2_path, output_file):
    with h5py.File(file1_path, 'r') as file1, h5py.File(file2_path, 'r') as file2, h5py.File(output_file, 'w') as output:

        # Read datasets from file 1
        chrom1 = file1['variants/CHROM'][:]
        pos1 = file1['variants/POS'][:]
        gt1 = file1['calldata/GT'][:]

        # Read datasets from file 2
        chrom2 = file2['variants/CHROM'][:]
        pos2 = file2['variants/POS'][:]
        gt2 = file2['calldata/GT'][:]

        # Concatenate datasets
        chrom_combined = np.concatenate((chrom1, chrom2))
        pos_combined = np.concatenate((pos1, pos2))
        gt_combined = np.concatenate((gt1, gt2))

        # Create groups and datasets in the output file with gzip compression
        variants_group = output.create_group('variants')
        variants_group.create_dataset('CHROM', data=chrom_combined, compression='gzip')
        variants_group.create_dataset('POS', data=pos_combined, compression='gzip')

        calldata_group = output.create_group('calldata')
        calldata_group.create_dataset('GT', data=gt_combined, compression='gzip')

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subset and concatenate HDF5 files")
    parser.add_argument('--subset', action='store_true', help="Subset HDF5 files")
    parser.add_argument('--concatenate', action='store_true', help="Concatenate HDF5 files")
    parser.add_argument('--h5py_file', type=str, help="Path to the HDF5 file for subsetting")
    parser.add_argument('--input_pass', type=str, help="Path to the pass variants file")
    parser.add_argument('--output_pass', type=str, help="Path to the output HDF5 file for subsetting")
    parser.add_argument('--input_populations', type=str, help="Path to the input populations file")
    parser.add_argument('--file1', type=str, help="Path to the first HDF5 file for concatenation")
    parser.add_argument('--file2', type=str, help="Path to the second HDF5 file for concatenation")
    parser.add_argument('--output_file', type=str, help="Path to the output HDF5 file for concatenation")

    args = parser.parse_args()

    if args.subset:
        if not all([args.h5py_file, args.input_pass, args.output_pass, args.input_populations]):
            print("For subsetting, you must provide --h5py_file, --input_pass, --output_pass, and --input_populations")
            sys.exit(1)
        subset_hdf5(args.h5py_file, args.input_pass, args.output_pass, args.input_populations)

    if args.concatenate:
        if not all([args.file1, args.file2, args.output_file]):
            print("For concatenation, you must provide --file1, --file2, and --output_file")
            sys.exit(1)
        concatenate_hdf5_files(args.file1, args.file2, args.output_file)
