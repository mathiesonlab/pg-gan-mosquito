"""
Plot model selection training process.
Author: Sara Mathieson
Date: 4/25/25
"""

# python imports
import matplotlib.pyplot as plt
import optparse
import sys

# TODO update library
def parse_args():
    """Parse command line arguments."""
    parser = optparse.OptionParser(description='plot model selection training')

    parser.add_option('-i', '--input', type='string', \
        help='path to input file of output results')
    parser.add_option('-o', '--output', type='string', \
        help='path to output figure (optional)')

    (opts, args) = parser.parse_args()

    mandatories = ['input']
    for m in mandatories:
        if not opts.__dict__[m]:
            print('mandatory option ' + m + ' is missing\n')
            parser.print_help()
            sys.exit()

    return opts

def parse_output(filename):
    f = open(filename,'r')

    train_acc_lst = []
    val_acc_lst = []

    for line in f:
        if line.startswith("Training acc over iter"):
            tokens = line.split(":")
            train_acc_lst.append(float(tokens[1]))
        
        elif line.startswith("Validation acc"):
            tokens = line.split(":")
            val_acc_lst.append(float(tokens[1]))

    f.close()

    # setup plot
    num_iter = len(train_acc_lst)
    assert num_iter == len(val_acc_lst)
    plt.plot(range(num_iter), train_acc_lst)
    plt.plot(range(num_iter), val_acc_lst)
    plt.plot(range(num_iter), [0.5]*num_iter, 'k--', lw=0.5)
    plt.legend(["train accuracy", "val accuracy"])
    plt.xlabel("training iteration")
    plt.ylabel("accuracy")
    plt.title("model selection training: dadi_joint vs. dadi_joint_mig")
    plt.show()

def main():
    opts = parse_args()
    parse_output(opts.input)

main()
