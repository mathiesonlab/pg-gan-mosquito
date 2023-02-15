"""
Prepare sample subsets (currently only gambiae)
Author: Sara Mathieson (based on R code from Matteo Fumagalli)
Date: 10/24/22
"""

import numpy as np
import optparse
import sys

"""Parse command line arguments."""
parser = optparse.OptionParser(description='prep gambiae samples')

parser.add_option('-i', '--input', type='string', \
    help='path to input file e.g. samples.species_aim.csv')
parser.add_option('-o', '--output', type='string', \
    help='path to output file e.g. AG1000G-BF-A_gamb_samples.txt')

(opts, args) = parser.parse_args()

mandatories = ['input', 'output']
for m in mandatories:
    if not opts.__dict__[m]:
        print('mandatory option ' + m + ' is missing\n')
        parser.print_help()
        sys.exit()

# species info for AG1000G-BF-A
speciesA = np.genfromtxt(opts.input, delimiter=',', skip_header=1, dtype="U")

# get indvs called as gambiae
ind_gam = speciesA[speciesA[:,4] == "gambiae"][:,0]

# write to a file for VCF prep
file_gam = open(opts.output,'w')
file_gam.write("\n".join(ind_gam) + "\n")
file_gam.close()
