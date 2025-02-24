# python imports
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

def main():
    input_file = sys.argv[1]
    output_run_plot = sys.argv[2]

    print("input file", input_file)
    print("training eval", output_run_plot)
    iter_stats_lst = parse_output(input_file)
    plot_output(iter_stats_lst, output_run_plot)


def parse_output(filename):
    """Parse pg-gan output to find the inferred parameters"""
    f = open(filename,'r')
    
    iter = None
    iter_stats_lst = []
    for line in f:
        line = line.strip()  # Remove trailing newline characters.
        if line.startswith("Start of iter"):
            if iter is None:
                # This only happens when the first line of the
                # FASTA file is parsed.
                iter = line.split(" ")[-1]
            else:
                #iter_stats = [iter, training_loss, validation_loss, train_acc, val_acc]
                iter_stats = [iter, training_loss, train_acc, val_acc]
                iter_stats = [ float(x) for x in iter_stats]
                iter_stats_lst.append(iter_stats)
                # Then reinitialise iter for new record
                iter = line.split(" ")[-1]
                
        elif line.startswith("Training loss"):
            training_loss = line.split(" ")[-1]
        elif line.startswith("Validation loss"):
            validation_loss = line.split(" ")[-1]
        elif line.startswith("Training acc over iter:"):
            train_acc = line.split(" ")[-1]
        elif line.startswith("Validation acc:"):
            val_acc = line.split(" ")[-1]
    return iter_stats_lst

def plot_output(iter_stats_lst, output_run_plot):
    # summarize history for accuracy
    x, training_loss, train_acc, val_acc = map(list, zip(*iter_stats_lst))
    
    fig, axs = plt.subplots(2)
    axs[0].plot(x, train_acc)
    axs[0].plot(x, val_acc)
    axs[0].set_ylabel('accuracy')
    axs[0].set_xlabel('iter')
    axs[0].legend(['Train', 'Validation'], loc='upper left')
    # summarize history for loss
    axs[1].plot(x, training_loss)
    #axs[1].plot(x, validation_loss)
    axs[1].set_ylabel('loss')
    axs[1].set_xlabel('iter')
    axs[1].legend(['Train'], loc='upper left')
    fig.tight_layout()
    plt.savefig(output_run_plot, dpi=350)
    return

        
        
    
if __name__ == "__main__":
    main()