# python imports
import datetime
import time
import numpy as np
import sys
import tensorflow as tf
import scipy.stats
import os

# our imports
import pg_gan
import global_vars
import util

def main():
    #in_file_data_1 = {'model': 'dadi_joint', 'params': 'NI,TG,NF,TS,NI1,NI2,NF1,NF2', 'data_h5': None, 'bed_file': None, 'reco_folder': None, 'grid': None, 'toy': None, 'seed': 1833, 'sample_sizes': '98,98', 'param_values': None}
    in_file_data_1 = {'model': 'dadi_joint_mig', 'params': 'NI,TG,NF,TS,NI1,NI2,NF1,NF2,MG', 'data_h5': None, 'bed_file': None, 'reco_folder': None, 'grid': None, 'toy': None, 'seed': 1833, 'sample_sizes': '98,98', 'param_values': None}
    in_file_data_2 = {'model': 'dadi_joint_mig', 'params': 'NI,TG,NF,TS,NI1,NI2,NF1,NF2,MG', 'data_h5': None, 'bed_file': None, 'reco_folder': None, 'grid': None, 'toy': None, 'seed': 1833, 'sample_sizes': '98,98', 'param_values': None}
    in_file_data_3 = {'model': 'dadi_joint_mig', 'params': 'NI,TG,NF,TS,NI1,NI2,NF1,NF2,MG', 'data_h5': None, 'bed_file': None, 'reco_folder': None, 'grid': None, 'toy': None, 'seed': 1833, 'sample_sizes': '98,98', 'param_values': None}
    print(in_file_data_1)
    print(in_file_data_2)
    #param_values_1 = DADI_PARAMS = 
    param_values_1 = DADI_MIG_PARAMS = [415254, 93341, 8292759, 11637, 2635696, 2748423, 11101754, 11439976, 120]    
    param_values_2 = DADI_MIG_PARAMS = [415254, 93341, 8292759, 11637, 2635696, 2748423, 11101754, 11439976, 0]
    param_values_3 = DADI_MIG_PARAMS = [415254, 93341, 8292759, 11637, 2635696, 2748423, 11101754, 11439976, 60]
    opts_1, posterior_1 = util.parse_args(in_file_data = in_file_data_1, param_values=param_values_1)
    opts_2, posterior_2 = util.parse_args(in_file_data = in_file_data_2, param_values=param_values_2)
    opts_3, posterior_3 = util.parse_args(in_file_data = in_file_data_3, param_values=param_values_3)
    real_data = None
    #demographic_model_selection([opts_1, opts_2], [posterior_1, posterior_2], real_data)
    demographic_model_selection([opts_1, opts_2, opts_3], [posterior_1, posterior_2, posterior_3], real_data)


################################################################################
# MODEL SELECTION
################################################################################
def demographic_model_selection(opts, posteriors, data_h5):
    '''
        cnn network following framework from https://keras.io/guides/writing_a_training_loop_from_scratch/
    '''
    #binary thus far
    num_batch = 5000
    mini_batch = 50
    #or call only generator like end of generator.py
    model_selection = MODEL_SELECTION(opts, posteriors)
    print("compiled generators")
    
    model_selection.build_generators_disc()
    if real_data:
        model_selection.build_iterator(data_h5)
    print(model_selection.opts)
    print(model_selection.posteriors)
    print(len(model_selection.generators))
    print(model_selection.disc)
    #initiate training loop
    iter = 50
    for iter in range(iter):
        print("\nStart of iter %d" % (iter,))
        print("time", datetime.datetime.now().time())
        start_time = time.time()
        #generate haplotype alignment images
        #5000 * 2 = 10000 images = 10000 / 50 = 200 stepsdisc
        x = model_selection.simulate_haplotype_alignments(num_batch)
        #formatting train and val_dataset   
        train_dataset, val_dataset = model_selection.prepare_dataset(x, num_batch, mini_batch)
        # Iterate over the batches of the created batch.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value = model_selection.train_step(x_batch_train, y_batch_train)
        # Log every 100 batches.
            if step != 0 and step % 50 == 0:
                print(
                    "Training loss (for one mini batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * mini_batch))

        train_acc = model_selection.train_acc_metric.result()
        print("Training acc over iter: %.4f" % (float(train_acc),))

        model_selection.train_acc_metric.reset_states()

        for x_batch_val, y_batch_val in val_dataset:
            model_selection.test_step(x_batch_val, y_batch_val)
        val_acc = model_selection.val_acc_metric.result()
        model_selection.val_acc_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Time taken: %.2fs" % (time.time() - start_time))
        sys.stdout.flush()
    print("finish loop")
    
    if real_data:
        pred_logits = model_selection.predict(num_batch = 10000)
        print(pred_logits)
        
        
    return

class MODEL_SELECTION:
    def __init__(self, opts, posteriors):
        self.opts = opts
        self.posteriors = posteriors
        self.disc = None
        self.generators = []
        self.iterator = None
        
        self.loss_fn= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        self.val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    def build_generators_disc(self):
        for opt, posteriors in zip(self.opts, self.posteriors):
            #TODO: only get generator
            #TODO:assert sample size all same
            generator, iterator, parameters, sample_sizes = util.process_opts(opt)
            generator.update_params(posteriors)
            self.generators.append(generator)
            
        self.disc = pg_gan.get_discriminator(sample_sizes)
        self.disc.dense3 = tf.keras.layers.Dense(len(self.opts))
        #TEMP
        self.disc.dropout.rate = 0
        return
    
    def build_iterator(self, data_h5):
        self.iterator = util.real_data_random.RealDataRandomIterator(data_h5, bed)
        return
    
    def simulate_haplotype_alignments(self, num_batch):
        assert len(self.posteriors) == len(self.generators)
        haplotype_alignments = [generator.simulate_batch(\
            params=posterior, batch_size = num_batch) for \
                generator, posterior in zip(self.generators, self.posteriors)]
        
        haplotype_alignments = np.concatenate(haplotype_alignments, axis=0)
        return haplotype_alignments

    def prepare_dataset(self, haplotype_alignments, num_batch, mini_batch):
        num_category = len(self.generators)
        y = [num for num in range(num_category) for _ in range(num_batch)]
        y = np.array(y).astype('float32').reshape((-1,1))
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
        self.train_acc_metric.update_state(y, logits)
        return loss_value

    @tf.function
    def test_step(self, x, y):
        val_logits = self.disc(x, training=False)
        self.val_acc_metric.update_state(y, val_logits)


    def predict(self, num_batch):
        real_regions = self.iterator.real_batch(neg1 = True, batch_size=num_batch)
        #process images
        logits = self.disc(real_regions, training=False)
        return logits


def get_dataset_partitions_tf(ds, ds_size, train_split=0.8, val_split=0.2,  shuffle=True, shuffle_size=10000):
    assert (train_split + val_split) == 1
    
    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    
    return train_ds, val_ds

    

if __name__ == "__main__":
    main()
