from meta_matching_network import *
from experiment_builder import ExperimentBuilder
import tensorflow.contrib.slim as slim
import data as dataset
import tqdm
from storage import save_statistics
import tensorflow as tf

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--way', type=int, default=5,
                        help='classes per set (default: 5)')
    parser.add_argument('--shot', type=int, default=1,
                        help='samples per class (default: 1)')
    parser.add_argument('--is_test', type=bool, default=False, help="Select FALSE for training, and True for testing")
    parser.add_argument('--ckp', type=int, default=-1,
                        help='Select corresponding checkpoint for testing (default: -1)')
    opt = parser.parse_args()
    print(opt)


    tf.reset_default_graph()
    # Experiment Setup
    sp = 1 # split
    batch_size = int(32 // sp) #  default 32 for 5way1shot
    classes_per_set = opt.way #20
    samples_per_class = opt.shot
    # N-way, K-shot
    continue_from_epoch = opt.ckp  # use -1 to start from scratch
    logs_path = "one_shot_outputs/"
    experiment_name = "LGM-Net_{}way{}shot".format(classes_per_set, samples_per_class)

    # Experiment builder
    data = dataset.MiniImageNetDataSet(batch_size=batch_size, classes_per_set=classes_per_set,
                                       samples_per_class=samples_per_class, shuffle_classes=True)
    experiment = ExperimentBuilder(data)
    one_shot_miniImagenet, losses, c_error_opt_op, init = experiment.build_experiment(batch_size,
                                                                                         classes_per_set,
                                                                                         samples_per_class)
    total_epochs = 120
    total_train_batches = 1000
    total_val_batches = int(250 * sp)
    total_test_batches = int(250 * sp)


    logs="{}way{}shot learning problems, with {} tasks per task batch".format(classes_per_set, samples_per_class, batch_size)
    save_statistics(experiment_name, ["Experimental details: {}".format(logs)])
    save_statistics(experiment_name, ["epoch", "train_c_loss", "train_c_accuracy", "val_loss", "val_accuracy",
                                      "test_c_loss", "test_c_accuracy", "learning_rate"])


    # Experiment initialization and running
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = "0"
    with tf.Session(config=config) as sess:
        sess.run(init)
        saver = tf.train.Saver(max_to_keep=5)
        if continue_from_epoch != -1: #load checkpoint if needed
            print("Loading from checkpoint")
            checkpoint = "saved_models/{}_{}.ckpt".format(experiment_name, continue_from_epoch)
            variables_to_restore = []
            tf.logging.info("The variables to restore")
            for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                print(var.name, var.get_shape())
                variables_to_restore.append(var)

            tf.logging.info('Fine-tuning from %s' % checkpoint)
            fine_tune = slim.assign_from_checkpoint_fn(checkpoint, variables_to_restore, ignore_missing_vars=True)
            fine_tune(sess)

        if opt.is_test:
            total_val_c_loss, total_val_accuracy = experiment.run_validation_epoch(total_val_batches=total_val_batches, sess=sess)
            print("Validating : val_loss: {}, val_accuracy: {}".format(total_val_c_loss, total_val_accuracy))
            total_test_c_loss, total_test_accuracy = experiment.run_testing_epoch(total_test_batches=total_test_batches, sess=sess)
            print("Testing: test_loss: {}, test_accuracy: {}".format(total_test_c_loss, total_test_accuracy))
            total_sg_test_accuracy, total_es_test_accuracy = experiment.run_ensemble_testing_epoch(total_test_batches=total_test_batches, sess=sess)
            print("Testing Ensemble: single accuracy {}, ensemble accuracy: {}".format(total_sg_test_accuracy, total_es_test_accuracy))
        else:
            with tqdm.tqdm(total=total_epochs) as pbar_e:
                for e in range(0, total_epochs):
                    total_c_loss, total_accuracy, lr = experiment.run_training_epoch(total_train_batches=total_train_batches,sess=sess)
                    print("Epoch {}: train_loss: {}, train_accuracy: {}".format(e, total_c_loss, total_accuracy))

                    total_val_c_loss, total_val_accuracy = experiment.run_validation_epoch(total_val_batches=total_val_batches, sess=sess)
                    print("Epoch {}: val_loss: {}, val_accuracy: {}".format(e, total_val_c_loss, total_val_accuracy))

                    total_test_c_loss, total_test_accuracy = experiment.run_testing_epoch(total_test_batches=total_test_batches, sess=sess)
                    print("Epoch {}: test_loss: {}, test_accuracy: {}".format(e, total_test_c_loss, total_test_accuracy))

                    save_statistics(experiment_name, [e, total_c_loss, total_accuracy, total_val_c_loss, total_accuracy,
                                                      total_test_c_loss, total_test_accuracy, 'lr: {}'.format(lr)])

                    save_path = saver.save(sess, "saved_models/{}_{}.ckpt".format(experiment_name, e))
                    pbar_e.update(1)

