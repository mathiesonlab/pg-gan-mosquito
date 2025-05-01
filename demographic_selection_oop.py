# python imports
import datetime
import time
import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras.optimizers import AdamW
import scipy.stats
import os
from sklearn.metrics import confusion_matrix
import argparse

# our imports
import pg_gan
import global_vars
import util

NUM_ITERS = 200

#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def main():
    parser = argparse.ArgumentParser(description='demographic_selection entry point')
    parser.add_argument('-b', '--working_dir', type=str, help='directory storing inputs and outputs')
    parser.add_argument('-d', '--data_h5', type=str, help='real data file')
    parser.add_argument("-s", action="store_true", dest="save_disc", default=False, help='save trained discriminator')
    parser.add_argument('-l', action="store_true", dest="load_disc", default=False, help='load discriminator and predict')
    parser.add_argument('-t', action="store_true", dest="toy", help='toy example')
    
    opts = parser.parse_args()
    
    #this is neccessary as reading the out files later will overwrite the arguments 
    opts_dict = vars(opts)
    print(opts_dict)
    #reset system argument so that it does not contradict with util.parse_args()
    sys.argv = [sys.argv[0]]

    #calls the desired posteriors derived from pg_gan run
    model_posteriors_file = open(os.path.join(opts_dict["working_dir"], 'input_posteriors.txt'), "r")
    model_posteriors = model_posteriors_file.read().strip("\n").split("\n")
    model_posteriors_file.close()
    
    print("model_posteriors")
    print(model_posteriors)
    
    #parse model_posteriors_file into respective opts and posterior values
    posterior_opts, posteriors= [], []
    for input_file in model_posteriors:
        posterior_param_values, in_file_data = parse_output(input_file, return_acc=False)
        print("in_file_data")
        print(in_file_data)
        print("posterior_param_values")
        print(posterior_param_values)
        #this line is needed to collect parser object from util.parse_args function
        opts, param_values = util.parse_args(in_file_data = in_file_data, param_values=posterior_param_values)
        posterior_opts.append(opts)
        posteriors.append(posterior_param_values)
        print("opts")
        print(opts)
        
       
    demographic_model_selection(posterior_opts, posteriors, work_dir = opts_dict["working_dir"], 
                                data_h5 = opts_dict["data_h5"], load_disc = opts_dict["load_disc"], 
                                save_disc = opts_dict["save_disc"], toy = opts_dict["toy"])
    


################################################################################
# MODEL SELECTION
################################################################################
def demographic_model_selection(opts, posteriors, work_dir, data_h5 = None, load_disc = False, save_disc = False, toy = False):
    '''
        cnn network following framework from https://keras.io/guides/writing_a_training_loop_from_scratch/
    '''
    if toy:
        print("toy run")
        iters = 2
        #override load trained disc and save disc option
        load_disc, save_disc = False, False
        
    else:  
        iters = NUM_ITERS
    
    num_batch = 500
    mini_batch = 50
    
    print('entering MODEL_SELECTION')
    model_selection = MODEL_SELECTION(opts, posteriors)
    model_selection.build_generators_disc()
    
    if data_h5:
        model_selection.build_iterator(data_h5, bed = None)
        print(model_selection.iterator)
            
    
    print(model_selection.opts)
    print(model_selection.posteriors)
    print(model_selection.disc)
    print("learning rate", model_selection.optimizer.learning_rate)
    sys.stdout.flush()
    
    disc_path = os.path.join(work_dir, 'trained_disc')
    
    if load_disc:
        try:            
            model_selection.disc = tf.keras.models.load_model(disc_path)
            print("loading trained classification model")
        except FileNotFoundError:
            print("No saved discriminator model found in directory")


    #initiate training loop
    else:
        for iter in range(iters):
            print("\nStart of iter %d" % (iter,))
            print("time", datetime.datetime.now().time())
            start_time = time.time()
            #generate haplotype alignment images
            x = model_selection.simulate_haplotype_alignments(num_batch)
            #generate labels
            y = model_selection.generate_labels(num_batch)
            #formatting train and val_dataset   
            train_dataset, val_dataset = model_selection.prepare_dataset(y, x, mini_batch)
            # Iterate over the batches of the created batch.
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                loss_value = model_selection.train_step(x_batch_train, y_batch_train)
            # Log every 50 batches.
                if step != 0 and step % 50 == 0:
                    print(
                        "Training loss (for one mini batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                    print("Seen so far: %s samples" % ((step + 1) * mini_batch))

            train_acc = model_selection.train_acc_metric.result()
            print("Training acc over iter: %.4f" % (float(train_acc),))

            model_selection.train_acc_metric.reset_state()

            for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
                print("val step", step)
                loss_value = model_selection.test_step(x_batch_val, y_batch_val)
                if step != 0 and step % 10 == 0:
                    print(
                        "Validation loss (for oneposterior_param_values mini batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                    print("Seen so far: %s samples" % ((step + 1) * mini_batch))

            val_acc = model_selection.val_acc_metric.result()
            model_selection.val_acc_metric.reset_state()
            print("Validation acc: %.4f" % (float(val_acc),))
            print("Time taken: %.2fs" % (time.time() - start_time))
            sys.stdout.flush()
            
        if save_disc:
            if not os.path.exists(disc_path):
                os.makedirs(disc_path)
            model_selection.disc.save(disc_path + "/mymodel.keras") # maybe add .keras here?
        
    #final evaluation and confusion metric
    num_test = 500 # reduced from 5000 due to memory issues
    x = model_selection.simulate_haplotype_alignments(num_test)
    y = model_selection.generate_labels(num_test)
    y_pred_logits = model_selection.predict(x)
    y_pred_softmax = tf.nn.softmax(y_pred_logits).numpy()
    y_pred_labels = np.argmax (y_pred_softmax, axis = 1)
    eval_confusion_matrix = confusion_matrix(y, y_pred_labels , normalize='pred')
    print("confusion matrix")
    print(eval_confusion_matrix)
    #output posterior's cnn probabilities for downstream ABC analysis
    # np.savetxt(os.path.join(work_dir, 'testSet_Predictions.txt'), y_pred_softmax)
    # np.savetxt(os.path.join(work_dir, 'testSet_labels.txt') , y_pred_labels)
    
    if data_h5:
        num_real = 1000
        # SM: changed neg1 to True to match sims
        data_h5_haplotype_alignments = model_selection.iterator.real_batch(neg1 = True, batch_size=num_real)
        y_pred_logits = model_selection.predict(data_h5_haplotype_alignments)
        y_pred_softmax = tf.nn.softmax(y_pred_logits).numpy()
        y_pred_softmax_arg_max = np.argmax(y_pred_softmax, axis = 1)
        print("percentage of images classified as class 0")
        print(int(np.count_nonzero(y_pred_softmax_arg_max == 0)) / num_real)
        print("percentage of images classified as class 1")
        print(int(np.count_nonzero(y_pred_softmax_arg_max == 1)) / num_real)
        #output real data's cnn probabilities for downstream ABC analysis
        np.savetxt(os.path.join(work_dir, 'Emp_Predictions.txt'), y_pred_softmax)
        
    return

class MODEL_SELECTION:
    def __init__(self, opts, posteriors):
        print("starting MODEL_SELECTION constructor")
        self.opts = opts
        print("after opts")
        self.posteriors = posteriors
        print("after post")
        self.disc = None
        print("after disc")
        self.generators = []
        print("after gen")
        #real data generator
        self.iterator = None
        print("after iterator")
        # SM: changing all 3 below from SparseCategorical to Binary
        self.loss_fn= tf.keras.losses.BinaryCrossentropy(from_logits=True)
        print("after loss")
        self.optimizer=AdamW(learning_rate=1e-4)
        print("after optimizer")
        self.train_acc_metric = tf.keras.metrics.BinaryAccuracy()
        print("after train metric")
        self.val_acc_metric = tf.keras.metrics.BinaryAccuracy()
        print("after val metric")

        print("ending MODEL_SELECTION constructor")

    def build_generators_disc(self):
        for opt, posteriors in zip(self.opts, self.posteriors):
            #TODO: only get generator
            #TODO:assert sample size all same
            generator, iterator, parameters, sample_sizes = util.process_opts(opt)
            generator.update_params(posteriors)
            self.generators.append(generator)
            #print(generator)
            #input('enter')
            
        print('sample sizes', sample_sizes)
        self.disc = pg_gan.get_discriminator(sample_sizes) # TODO why FC part?
        # SM: removing all this for now...
        #self.disc.fc1 = tf.keras.layers.Dense(256, activation='relu')
        #self.disc.fc2 = tf.keras.layers.Dense(256, activation='relu')
        #self.disc.dense3 = tf.keras.layers.Dense(len(self.opts))
        #TEMP
        #self.disc.dropout.rate = 0
        return
    
    def build_iterator(self, data_h5, bed):
        self.iterator = util.real_data_random.RealDataRandomIterator(data_h5, bed)
        return
    
    def simulate_haplotype_alignments(self, num_batch):
        assert len(self.posteriors) == len(self.generators)
        haplotype_alignments = [generator.simulate_batch(\
            params=posterior, batch_size = num_batch) for \
                generator, posterior in zip(self.generators, self.posteriors)]
        
        #print(haplotype_alignments)
        #input('enter')
        haplotype_alignments = np.concatenate(haplotype_alignments, axis=0)
        return haplotype_alignments
    
    def generate_labels(self, num_batch):
        num_category = len(self.generators)
        y = [num for num in range(num_category) for _ in range(num_batch)]
        y = np.array(y).astype('float32').reshape((-1,1))
        return y

    def prepare_dataset(self, y, haplotype_alignments, mini_batch):
        dataset = tf.data.Dataset.from_tensor_slices((haplotype_alignments, y))
        train_dataset, val_dataset = get_dataset_partitions_tf(dataset, int(tf.data.experimental.cardinality(dataset)))
        # Prepare the training dataset for shuffling.
        train_dataset = train_dataset.batch(mini_batch)
        val_dataset = val_dataset.batch(mini_batch)
        return train_dataset, val_dataset

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.disc(x, training=True)
            loss_value = self.loss_fn(y, logits)
        grads = tape.gradient(loss_value, self.disc.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.disc.trainable_weights))
        tf.print(tf.reduce_sum(y))
        self.train_acc_metric.update_state(y, logits)
        return loss_value

    @tf.function
    def test_step(self, x, y):
        val_logits = self.disc(x, training=False)
        loss_value = self.loss_fn(y, val_logits)
        #print("testing", len(y), len(val_logits))
        '''for i in range(len(y)):
            if y[i][0] == 1 and val_logits[i][0] > 0:
                print("correct!", y[i][0], val_logits[i][0])
            elif y[i][0] == 0 and val_logits[i][0] < 0:
                print("correct!", y[i][0], val_logits[i][0])
            #tf.print(val_logits[i][0])'''
        #print("num 1s")
        #tf.print(tf.reduce_sum(y))
        #tf.print(tf.nn.sigmoid(val_logits))
        #print(y.numpy())
        #print(val_logits.numpy())
        self.val_acc_metric.update_state(y, tf.nn.sigmoid(val_logits))
        return loss_value


    def predict(self, haplotype_alignments):
        #process images
        logits = self.disc(haplotype_alignments, training=False)
        return logits


def get_dataset_partitions_tf(ds, ds_size, train_split=0.8, val_split=0.2,  shuffle=True, shuffle_size=100): # SM: 10000 -> 100 due to OOM error on GPU
    assert (train_split + val_split) == 1
    
    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    print("train size, val size", train_size, val_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    
    return train_ds, val_ds

###############################################################################################
#TEMP due to seperated environment
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
    
def parse_args(in_file_data = None, param_values = None):
    """require parser object but """
    parser = optparse.OptionParser(description='PG-GAN entry point')

    parser.add_option('-m', '--model', type='string',help='exp, im, ooa2, ooa3')
    parser.add_option('-p', '--params', type='string',
        help='comma separated parameter list')
    parser.add_option('-d', '--data_h5', type='string', help='real data file')
    parser.add_option('-b', '--bed', type='string', help='bed file (mask)')
    parser.add_option('-r', '--reco_folder', type='string',
        help='recombination maps')
    parser.add_option('-g', action="store_true", dest="grid",help='grid search')
    parser.add_option('-t', action="store_true", dest="toy", help='toy example')
    parser.add_option('-s', '--seed', type='int', default=1833,
        help='seed for RNG')
    parser.add_option('-n', '--sample_sizes', type='string',
        help='comma separated sample sizes for each population, in haps')
    parser.add_option('-v', '--param_values', type='string',
        help='comma separated values corresponding to params')

    print(parser)
    (opts, args) = parser.parse_args()

    '''
    The following section overrides params from the input file with the provided
    args.
    '''

    # note: this series of checks looks like it could be simplified with list
    #       iteration:
    # it can't be, bc the opts object can't be indexed--eg opts['model'] fails
    
    def param_mismatch(param, og, replacement):
        print("***** WARNING: MISMATCH BETWEEN IN FILE AND CMD ARGS: " + param +
              ", using ARGS (" + str(og) + " -> " + str(replacement) + ")")
    # because we care about the seed from the trial, here in_file_data takes over opts
    if in_file_data is not None:
        if opts.model is None:
            opts.model = in_file_data['model']

        if opts.params is None:
            opts.params = in_file_data['params']

        if opts.data_h5 is None:
            opts.data_h5 = in_file_data['data_h5']

        if opts.bed is None:
            opts.bed = in_file_data['bed_file']

        if opts.sample_sizes is None:
            opts.sample_sizes = in_file_data['sample_sizes']

        if opts.reco_folder is None:
            opts.reco_folder = in_file_data['reco_folder']

        if in_file_data['seed'] is not None:
            opts.seed = in_file_data['seed']
            
    if opts.param_values is not None:
        arg_values = [float(val_str) for val_str in
            opts.param_values.split(',')]
        if arg_values != param_values:
            param_mismatch("PARAM_VALUES", param_values, arg_values)
            param_values = arg_values # override at return

    mandatories = ['model','params']
    for m in mandatories:
        if not opts.__dict__[m]:
            print('mandatory option ' + m + ' is missing\n')
            parser.print_help()
            sys.exit()

    if param_values is None:
        return opts

    return opts, param_values
##############################################################################################################

if __name__ == "__main__":
    main()
