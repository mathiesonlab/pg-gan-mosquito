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
    in_file_data_1 = {'model': 'dadi_joint', 'params': 'NI,TG,NF,TS,NI1,NI2,NF1,NF2', 'data_h5': None, 'bed_file': None, 'reco_folder': None, 'grid': None, 'toy': None, 'seed': 1833, 'sample_sizes': '98,98', 'param_values': None}
    in_file_data_2 = {'model': 'dadi_joint_mig', 'params': 'NI,TG,NF,TS,NI1,NI2,NF1,NF2,MG', 'data_h5': None, 'bed_file': None, 'reco_folder': None, 'grid': None, 'toy': None, 'seed': 1833, 'sample_sizes': '98,98', 'param_values': None}
    print(in_file_data_1)
    print(in_file_data_2)
    param_values_1 = DADI_PARAMS = [420646, 89506, 9440437, 2245, 18328570, 42062652, 42064645, 42064198]
    param_values_2 = DADI_MIG_PARAMS = [415254, 93341, 8292759, 11637, 2635696, 2748423, 11101754, 11439976, 40]
    opts_1, posterior_1 = util.parse_args(in_file_data = in_file_data_1, param_values=param_values_1)
    opts_2, posterior_2 = util.parse_args(in_file_data = in_file_data_2, param_values=param_values_2)
    real_data = None
    demographic_model_selection(opts_1, opts_2, posterior_1, posterior_2, real_data)

################################################################################
# MODEL SELECTION
################################################################################
def demographic_model_selection(opts_1, opts_2, posterior_1, posterior_2, real_data):
    '''
        cnn network following framework from https://keras.io/guides/writing_a_training_loop_from_scratch/
    '''
    #binary thus far
    mini_batch = 50
    #or call only generator like end of generator.py
    generator_1, iterator, parameters_1, sample_sizes_1 = util.process_opts(opts_1)
    generator_2, iterator, parameters_2, sample_sizes_2 = util.process_opts(opts_2)
    assert sample_sizes_1 == sample_sizes_2, ('opts sample sizes ought to be the same')
    print("compiled generators")

    #iterator = util.real_data_random.RealDataRandomIterator(real_data, opts_1.bed)

    #update with posterior parameters
    # generator_1.update_params(posterior_1)
    # generator_2.update_params(posterior_2)

    disc = pg_gan.get_discriminator(sample_sizes_1)
    loss_fn= tf.keras.losses.CategoricalCrossentropy()
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)
    print("compiled discriminator")

    train_acc_metric = tf.keras.metrics.CategoricalCrossentropy()
    val_acc_metric = tf.keras.metrics.CategoricalCrossentropy()

    #initiate training loopdatetime.datetime.now().time()
    epochs = 100
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        print("time", datetime.datetime.now().time())
        start_time = time.time()

        #generate haplotype alignment images
        generated_regions_1 = generator_1.simulate_batch(params=posterior_1, batch_size = 500)
        generated_regions_2 = generator_2.simulate_batch(params=posterior_2, batch_size = 500)

        #formatting train and val_dataset
        x = np.concatenate([generated_regions_1, generated_regions_2], axis=0)
        y = [1] * generated_regions_1.shape[0] + [0] * generated_regions_2.shape[0]
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        train_dataset, val_dataset = get_dataset_partitions_tf(dataset, int(tf.data.experimental.cardinality(dataset)))
        # Prepare the training dataset for shuffling.
        train_dataset = train_dataset.batch(mini_batch)
        val_dataset = val_dataset.batch(mini_batch)
        sys.stdout.flush()
        # Iterate over the batches of the created batch.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = disc(x_batch_train, training=True)  # Logits for this minibatch
                # Compute the loss value for this minibatch.
                loss_value = loss_fn(y_batch_train, logits)
            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, disc.trainable_weights)

            # Run one step of gradient descent by updating the value of the variables to minimize the loss.
            #trainable_weights identical as trainable_variables
            optimizer.apply_gradients(zip(grads, disc.trainable_weights))

            train_acc_metric.update_state(y_batch_train, logits)

        # Log every 100 batches.
            if step % 50 == 0:
                print(
                    "Training loss (for one mini batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * mini_batch))

        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        train_acc_metric.reset_states()

        for x_batch_val, y_batch_val in val_dataset:
            val_logits = disc(x_batch_val, training=False)
            # Update val metrics
            val_acc_metric.update_state(y_batch_val, val_logits)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Time taken: %.2fs" % (time.time() - start_time))

    return

class MODEL_SELECTION:
    def __init__(self, generator, parameters):
        self.generator_1 = generator_1
        self.generator_2 = generator_2
        self.CNN = disc
        self.iterator = iterator
        self.parameters_1 = parameters_1
        self.parameters_2 = parameters_2

    def dataset_simulator(self):
        """
            simulator: function that returns (data, label) tuples, where data is a <input_shape> and label is an <output_shape> shaped numpy array

        """
        generated_regions_1 = self.generator.simulate_batch(params=self.parameter_1)
        generated_regions_2 = self.generator.simulate_batch(params=self.parameter_2)
        #join and shuffle 
        train_ds = tf.data.Dataset.from_tensor_slices((filenames, labels))
        return train_ds

    def train(self, num_batches,train_ds):
        self.CNN.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        model.fit(train_ds, epochs=num_batches)



    def predict(self):
        real_regions = self.iterator.real_batch(neg1 = True)
        #process images
        predictions = model.predict(img_array)
        return scores


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
