import tensorflow as tf
import tqdm
from meta_matching_network import MetaMatchingNetwork
from scipy.stats import mode
import numpy as np

class ExperimentBuilder:

    def __init__(self, data):
        """
        Initializes an ExperimentBuilder object. The ExperimentBuilder object takes care of setting up our experiment
        and provides helper functions such as run_training_epoch and run_validation_epoch to simplify out training
        and evaluation procedures.
        :param data: A data provider class
        """
        self.data = data

    def build_experiment(self, batch_size, classes_per_set, samples_per_class, init_lr = 1e-3):
        """
        :param batch_size: The experiment batch size
        :param classes_per_set: An integer indicating the number of classes per support set
        :param samples_per_class: An integer indicating the number of samples per class
        :param init_lr: The initial learning rate
        :return: some ops
        """

        height, width, channels = self.data.x_train.shape[2], self.data.x_train.shape[3], self.data.x_train.shape[4] # (84, 84, 3)

        ## Construct placeholders
        self.support_set_images = tf.placeholder(tf.float32, [batch_size, classes_per_set, samples_per_class, height, width,
                                                              channels], 'support_set_images')
        self.support_set_labels = tf.placeholder(tf.int32, [batch_size, classes_per_set, samples_per_class], 'support_set_labels')
        self.target_image = tf.placeholder(tf.float32, [batch_size, height, width, channels], 'target_image')
        self.target_label = tf.placeholder(tf.int32, [batch_size], 'target_label')
        self.training_phase = tf.placeholder(tf.bool, name='training-flag')
        self.rotate_flag = tf.placeholder(tf.bool, name='rotate-flag')
        self.keep_prob = tf.placeholder(tf.float32, name='dropout-prob')
        self.current_learning_rate = init_lr # 1e-3
        self.learning_rate = tf.placeholder(tf.float32, name='learning-rate-set')

        ##
        self.one_shot_learner = MetaMatchingNetwork(batch_size=batch_size, support_set_images=self.support_set_images,
                                            support_set_labels=self.support_set_labels,
                                            target_image=self.target_image, target_label=self.target_label,
                                            keep_prob=self.keep_prob,
                                            is_training=self.training_phase, rotate_flag=self.rotate_flag,
                                            num_classes_per_set=classes_per_set,
                                            num_samples_per_class=samples_per_class, learning_rate=self.learning_rate)

        _, self.losses, self.c_error_opt_op = self.one_shot_learner.init_train()
        init = tf.global_variables_initializer()
        self.total_train_iter = 0
        return self.one_shot_learner, self.losses, self.c_error_opt_op, init

    def run_training_epoch(self, total_train_batches, sess):
        """
        Runs one training epoch
        :param total_train_batches: Number of batches to train on
        :param sess: Session object
        :return: mean_training_categorical_crossentropy_loss and mean_training_accuracy
        """
        total_c_loss = 0.
        total_accuracy = 0.
        with tqdm.tqdm(total=total_train_batches) as pbar:

            for i in range(total_train_batches):  # train epoch
                x_support_set, y_support_set, x_target, y_target = self.data.get_train_batch(augment=True)
                _, c_loss_value, acc = sess.run(
                    [self.c_error_opt_op, self.losses['losses'], self.losses['accuracy']],
                    feed_dict={self.keep_prob: 1.0, self.support_set_images: x_support_set,
                               self.support_set_labels: y_support_set, self.target_image: x_target, self.target_label: y_target,
                               self.training_phase: True, self.rotate_flag: False, self.learning_rate: self.current_learning_rate})

                iter_out = "train_loss: {}, train_accuracy: {}, current lr: {}".format(c_loss_value, acc, self.current_learning_rate)
                pbar.set_description(iter_out)

                pbar.update(1)
                total_c_loss += c_loss_value
                total_accuracy += acc
                self.total_train_iter += 1
                if self.total_train_iter % 1500 == 0:
                    self.current_learning_rate /= 1.11111
                    self.current_learning_rate = max(1e-6, self.current_learning_rate)
                    # set a lower bound of the learning rate, 1e-6 is reasonable, 1e-8 is too small
                    print("Change learning rate to ", self.current_learning_rate)

        total_c_loss = total_c_loss / total_train_batches
        total_accuracy = total_accuracy / total_train_batches
        return total_c_loss, total_accuracy, self.current_learning_rate

    # some check functions for debugging
    def run_check_gradient(self, sess):
        x_support_set, y_support_set, x_target, y_target = self.data.get_train_batch(augment=True)
        feed_dict = {self.keep_prob: 1.0, self.support_set_images: x_support_set,
                     self.support_set_labels: y_support_set, self.target_image: x_target, self.target_label: y_target,
                     self.training_phase: True, self.rotate_flag: False, self.learning_rate: self.current_learning_rate}
        self.one_shot_learner.check_gradients_magnitude(sess, feed_dict=feed_dict)

    def run_check_tensor(self, sess):
        x_support_set, y_support_set, x_target, y_target = self.data.get_train_batch(augment=True)
        feed_dict = {self.keep_prob: 1.0, self.support_set_images: x_support_set,
                     self.support_set_labels: y_support_set, self.target_image: x_target, self.target_label: y_target,
                     self.training_phase: True, self.rotate_flag: False, self.learning_rate: self.current_learning_rate}
        self.one_shot_learner.check_tensors_magnitude(sess, feed_dict=feed_dict)

    def run_check_g(self, sess):
        self.one_shot_learner.check_g(sess)


    def run_check_genweights(self, sess):
        """
        To generate the weights distribution of a task
        :param sess:
        :return:
        """
        x_support_set, y_support_set, x_target, y_target = self.data.get_test_batch(augment=True)

        emb_list = []
        label_list = []
        for i in range(10):
            feed_dict = {self.keep_prob: 1.0, self.support_set_images: x_support_set,
                         self.support_set_labels: y_support_set, self.target_image: x_target, self.target_label: y_target,
                         self.training_phase: False, self.rotate_flag: False, self.learning_rate: self.current_learning_rate}
            tasks_gen_weights_list = self.losses['tasks_gen_weights_list'] # shape is batchsize x 4( four tensor for batchsize tasks)
            tgws = sess.run(tasks_gen_weights_list, feed_dict=feed_dict)
            tw_np_list = []
            for tgw in tgws:
                tw = np.concatenate([tgw[0].reshape(-1), tgw[1].reshape(-1), tgw[2].reshape(-1), tgw[3].reshape(-1)])
                tw_np_list.append(tw)

            emb = np.array(tw_np_list)
            label = np.arange(len(tgws))
            emb_list.append(emb)
            label_list.append(label)

        embs = np.array(emb_list)
        labels = np.array(label_list)
        print("get result shape ", embs.shape, labels.shape)
        np.savez("data.npz", embs = embs, labels=labels)
        # todo better print some predictions results, to see the performance, we plot good predicted weights
        # tsne is not influenced by order, hence, here, we just give it a label, and store several inference


    def run_validation_epoch(self, total_val_batches, sess):
        """
        Runs one validation epoch
        :param total_val_batches: Number of batches to train on
        :param sess: Session object
        :return: mean_validation_categorical_crossentropy_loss and mean_validation_accuracy
        """
        total_val_c_loss = 0.
        total_val_accuracy = 0.

        with tqdm.tqdm(total=total_val_batches) as pbar:
            for i in range(total_val_batches):  # validation epoch
                x_support_set, y_support_set, x_target, y_target = self.data.get_val_batch(augment=True)
                c_loss_value, acc = sess.run(
                    [self.losses['losses'], self.losses['accuracy']],
                    feed_dict={self.keep_prob: 1.0, self.support_set_images: x_support_set,
                               self.support_set_labels: y_support_set, self.target_image: x_target, self.target_label: y_target,
                               self.training_phase: False, self.rotate_flag: False})

                iter_out = "val_loss: {}, val_accuracy: {}".format(c_loss_value, acc)
                pbar.set_description(iter_out)
                pbar.update(1)

                total_val_c_loss += c_loss_value
                total_val_accuracy += acc

        total_val_c_loss = total_val_c_loss / total_val_batches
        total_val_accuracy = total_val_accuracy / total_val_batches

        return total_val_c_loss, total_val_accuracy

    def run_testing_epoch(self, total_test_batches, sess):
        """
        Runs one testing epoch
        :param total_test_batches: Number of batches to train on
        :param sess: Session object
        :return: mean_testing_categorical_crossentropy_loss and mean_testing_accuracy
        """
        total_test_c_loss = 0.
        total_test_accuracy = 0.
        with tqdm.tqdm(total=total_test_batches) as pbar:
            for i in range(total_test_batches):
                x_support_set, y_support_set, x_target, y_target = self.data.get_test_batch(augment=True)
                c_loss_value, acc = sess.run(
                    [self.losses['losses'], self.losses['accuracy']],
                    feed_dict={self.keep_prob: 1.0, self.support_set_images: x_support_set,
                               self.support_set_labels: y_support_set, self.target_image: x_target,
                               self.target_label: y_target,
                               self.training_phase: False, self.rotate_flag: False})

                iter_out = "test_loss: {}, test_accuracy: {}".format(c_loss_value, acc)
                pbar.set_description(iter_out)
                pbar.update(1)

                total_test_c_loss += c_loss_value
                total_test_accuracy += acc
            total_test_c_loss = total_test_c_loss / total_test_batches
            total_test_accuracy = total_test_accuracy / total_test_batches
        return total_test_c_loss, total_test_accuracy


    def run_ensemble_testing_epoch(self, total_test_batches, sess):
        """
        :param total_test_batches:
        :param sess:
        :return:
        """
        print("********* run_ensemble_testing_epoch")
        total_es_test_accuracy = 0. # ensemble model
        total_sg_test_accuracy = 0. # single model
        n_ensembles = 10 # 20
        with tqdm.tqdm(total=total_test_batches) as pbar:
            for i in range(total_test_batches):
                x_support_set, y_support_set, x_target, y_target = self.data.get_test_batch(augment=True)
                t_list = []
                for i in range(n_ensembles):
                    preds = sess.run(self.losses['preds'], feed_dict={self.keep_prob: 1.0, self.support_set_images: x_support_set,
                                                                      self.support_set_labels: y_support_set,
                                                                      self.target_image: x_target,
                                                                      self.target_label: y_target,
                                                                      self.training_phase: False, self.rotate_flag: False})
                    t_preds = np.argmax(preds, axis=1)
                    t_list.append(t_preds)

                ens_preds_st = np.stack(t_list, axis=0)
                ens_preds = mode(ens_preds_st, axis=0)[0][0]
                one_acc = np.mean(t_preds==y_target)
                ens_acc = np.mean(ens_preds==y_target)

                # print("Ensemble prediction {}".format(ens_preds_st))
                # print("e {}".format(ens_preds))
                # print("y {}".format(y_target.astype(np.int32)))

                iter_out = "Ensemble test_accuracy: {}, single model test accuracy: {}".format(ens_acc, one_acc)
                # print(iter_out)
                pbar.set_description(iter_out)
                pbar.update(1)

                total_es_test_accuracy += ens_acc
                total_sg_test_accuracy += one_acc

            total_sg_test_accuracy = total_sg_test_accuracy / total_test_batches
            total_es_test_accuracy = total_es_test_accuracy / total_test_batches
        return total_sg_test_accuracy, total_es_test_accuracy
