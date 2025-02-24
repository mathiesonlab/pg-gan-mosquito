import h5py
import sys


def print_hdf5_tree(hdf5_file):
    def print_tree(name, obj):
        """ Helper function to print the tree structure, shape of datasets, and compression info in the HDF5 file. """
        indent = '    ' * (name.count('/') - 1)
        if isinstance(obj, h5py.Dataset):
            # Check if the dataset is compressed
            compression = obj.compression if obj.compression else 'None'
            # Print the name, shape, and compression type for datasets
            print(f"{indent}└── {name.split('/')[-1]} (shape: {obj.shape}, compression: {compression})")
        else:
            # Print the name for groups
            print(f"{indent}└── {name.split('/')[-1]}")

    with h5py.File(hdf5_file, 'r') as file:
        print(f"Tree structure of '{hdf5_file}':")
        file.visititems(print_tree)



# Usage
if __name__ == "__main__":
    print_hdf5_tree('ag1000g.phase2.ar1.haplotypes.3L.h5')
    print_hdf5_tree('ag1000g.phase2.ar1.haplotypes.3R.h5')
    print_hdf5_tree('/data/SBBS-FumagalliLab/mosquito_gan/pg-gan-mosquito/GNB-BFA_gamb_nsg.h5')
    
