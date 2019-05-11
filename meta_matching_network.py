"""
In this file, I reimplement the function, and change the APIs to make it flexible to change functions.
Since the whole architecture has clear architecture, so here it will not be too difficult.
"""
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.python.ops.nn_ops import max_pool, avg_pool
import numpy as np

def print_params(vars_list, name=None):
    print("#"*30)
    if name is not None:
        print("The variables of ", name)
    for var in vars_list:
        print(var.name, var.get_shape())
    print("#"*30)
def leaky_relu(x, leak=0.2, name='leaky_relu'):
    return tf.maximum(x, x * leak, name=name)
def relu(x, name='relu'):
    return tf.nn.relu(x, name=name)

def normalization(inputs, training, type='layer_norm'):
    """
    :param inputs:
    :param training:
    :param type: 'batch_norm' , 'instance_norm' , 'layer_norm'
    :return:
    """
    if type == 'batch_norm':
        return tf.contrib.layers.batch_norm(inputs, updates_collections=None, decay=0.99,
                                     scale=True, center=True, is_training=training)
    elif type == 'instance_norm':
        return tf.contrib.layers.instance_norm(inputs, center=True, scale=True)
    elif type == 'layer_norm':
        return tf.contrib.layers.layer_norm(inputs, center=True, scale=True)

class DistanceNetwork:
    def __init__(self, metric='cosine'):
        """
        :param metric: 'cosine', 'euclidean'
        'cosine' is better, but also can use 'euclidean
        """
        self.reuse = False
        self.metric = metric

    def __call__(self, support_set, input_image, name='_distance', training=False):
        """
        This module calculates the cosine distance between each of the support set embeddings and the target
        image embeddings.
        :param support_set: The embeddings of the support set images, tensor of shape [batch_size, spc, 64] (32,5,576)
        :param input_image: The embedding of the target image, tensor of shape [batch_size, 64] (32, 576)
        :param name: Name of the op to appear on the graph
        :param training: Flag indicating training or evaluation (True/False)
        :return: A tensor with cosine similarities of shape [batch_size, sequence_length, 1]
        """
        print("In DistanceNetwork, using ", self.metric)
        if self.metric == 'cosine':
            with tf.name_scope(self.metric+name):
                input_image = tf.expand_dims(input_image, axis=1)
                norm_s = tf.nn.l2_normalize(support_set, dim=2)
                norm_i = tf.nn.l2_normalize(input_image, dim=2)
                similarities = tf.reduce_sum(tf.multiply(norm_s, norm_i), axis=2)
        elif self.metric == 'euclidean':
            with tf.name_scope(self.metric + name):
                # euclidean distance should use negative one to be similarities, large distance means different
                input_image = tf.expand_dims(input_image, axis=1)
                similarities = -tf.reduce_mean(tf.square(support_set - input_image), axis=2)
        else:
            raise TypeError("Choose distance metrics from cosine, euclidean ")

        print("Similarities ", similarities)  # (32, 5)
        return similarities

class AttentionalClassify:
    def __init__(self):
        self.reuse = False

    def __call__(self, similarities, support_set_y, name, training=False):
        """
        Produces pdfs over the support set classes for the target set image.
        n*k is sequence length
        :param similarities: A tensor with cosine similarities of size [ batch_size, n*k] (32, 20)
        :param support_set_y: A tensor with the one hot vectors of the targets for each support set image
                                                                            [batch_size, n*k, num_classes] (32, nk, 5)
        :param name: The name of the op to appear on tf graph
        :param training: Flag indicating training or evaluation stage (True/False)
        :return: Softmax pdf
        """
        print("In AttentionalClassify")
        print(similarities.get_shape(), support_set_y.get_shape())
        with tf.name_scope('attentional-classification' + name), tf.variable_scope('attentional-classification',
                                                                                   reuse=self.reuse):
            softmax_similarities = tf.nn.softmax(similarities) # (32,5)
            preds = tf.squeeze(tf.matmul(tf.expand_dims(softmax_similarities, 1), support_set_y)) # (32,5)

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='attentional-classification')
        return preds

### Meta Network Classes

"""Fully Connected Meta Network, with reparameterization tricks """
class MetaNetwork:
    """
    A general version of Meta Network, which can generate weights for both MLP and CNN with bias...
    Without sharing parameters for each kernel; This will requires more meta weights
    """
    def __init__(self):
        self.reuse = False

    def __call__(self, inputs, context, out_size=64, kernel_size=[3, 3], name='Meta'):
        """
        :param inputs: (nk+1, 6,6,128) tensor containing all samples in a task
        :param context: (64,) context vector for a task
        :param out_size:
        :param kernel_size:
        :param name:
        :return:
        """
        print("In meta network, inputs shape:{}, context shape: {}".format(inputs.get_shape(), context.get_shape()))
        # (6, 11, 11, 128), context shape: (64,)

        inputs_shape_list = inputs.get_shape().as_list()
        c_dim = context.get_shape().as_list()[-1]
        # split the context into mean and variance predicted by task context encoder
        z_dim = c_dim // 2
        c_mu = context[:z_dim]
        c_log_var = context[z_dim:]


        if len(inputs_shape_list) == 4:
            is_CNN = True
        else:
            is_CNN = False

        if is_CNN == True:
            assert kernel_size[0] == kernel_size[1]
            f_size = kernel_size[0] # filter size
            in_size = inputs_shape_list[-1] # input channel number 64

            M = f_size*f_size*in_size
            N = out_size
            wt_shape = [M+1, N]  # weights tensor shape, with bias
        else:
            M = inputs_shape_list[-1]
            N = out_size
            wt_shape = [M+1, N]

        with tf.variable_scope("MetaNetwork_" + name, reuse=self.reuse):

            with tf.variable_scope("z_signal"):
                z_signal = tf.random_normal(shape=[1, z_dim], name='z_signal')

            # reparameterization trick
            z_c_mu = tf.expand_dims(c_mu, axis=0)
            z_c_log_var = tf.expand_dims(c_log_var, axis=0)
            print(z_c_mu.get_shape(), z_c_log_var.get_shape(), z_signal.get_shape())
            z_c = z_c_mu + tf.exp(z_c_log_var/2)*z_signal

            with tf.variable_scope("meta_weights"):
                w1 = tf.get_variable('w1', [z_dim, (M+1)*N], initializer=tf.glorot_uniform_initializer())
                b1 = tf.get_variable('b1', [(M+1)*N], initializer=tf.constant_initializer(0.0))
                final = tf.matmul(z_c, w1) + b1 # (N, M+1)
                meta_weights = final[0, :M*N]
                meta_bias = final[0, M*N:]
                print("Meta weights ", meta_weights.get_shape(), meta_bias.get_shape())

            if is_CNN:
                meta_weights = tf.transpose(tf.reshape(meta_weights, (out_size, in_size, f_size, f_size)))
            else:
                meta_weights = tf.transpose(tf.reshape(meta_weights, (out_size, M)))

            # print("meta weights ", meta_weights, meta_bias)
            with tf.variable_scope("normalize_weights"):
                if is_CNN:
                    meta_weights = tf.nn.l2_normalize(meta_weights, dim=[0, 1, 2]) # exp0
                else:
                    meta_weights = tf.nn.l2_normalize(meta_weights, dim=[0]) # exp0
        return meta_weights, meta_bias


class MetaConvolution:
    """
    Meta Convolutional Network
    """
    def __init__(self):
        self.reuse=False
        self.metanet = MetaNetwork()
    def __call__(self, inputs, context, filters, kernel_size, training=False, name='meta_conv'):
        """
        :param inputs: A convolutional Tensor (nk+1, 6, 6, 128)
        :param context: a vector represent corresponding task context, which is (64, ) tensor
        :param filters: meta network output channel number
        :param kernel_size:
        :param training:
        :param keep_prob: In fact, this is a placeholder
        :return:
        """
        # print("inputs ", inputs.get_shape())
        # print("context ", context.get_shape())
        meta_conv_w, meta_conv_b = self.metanet(inputs, context, out_size=filters, kernel_size=kernel_size, name=name)
        tf.add_to_collection('meta_conv_w', meta_conv_w)
        tf.add_to_collection('meta_conv_b', meta_conv_b)
        outputs = tf.nn.conv2d(inputs, meta_conv_w, strides=[1, 1, 1, 1], padding='SAME') + meta_conv_b
        self.reuse = True
        return outputs, meta_conv_w, meta_conv_b


### Task encoder Classes

class TaskTransformer:
    def __init__(self):
        self.reuse = False
        self.layer_sizes = [64, 64, 64] # 64->32
    def __call__(self, task_embedding, training=False, keep_prob=1.0):
        """

        :param task_images: images from a task
        :param training:
        :return:
        """
        with tf.variable_scope("TaskTransformer", reuse=self.reuse):
            # 11*11
            with tf.variable_scope('t_conv1'):
                te = tf.layers.conv2d(task_embedding, self.layer_sizes[0], [3, 3], strides=(1, 1),
                                                   padding='SAME')
                te = relu(te, name='relu')
                te = normalization(te, training=training, type='batch_norm')
                te = max_pool(te, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # 6*6
            with tf.variable_scope('t_conv2'):
                te = tf.layers.conv2d(te, self.layer_sizes[1], [3, 3], strides=(1, 1), padding='SAME')
                te = relu(te, name='relu')
                te = normalization(te, training=training, type='batch_norm')
                te = max_pool(te, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # 3*3
            with tf.variable_scope("t_conv3"):
                te = tf.layers.conv2d(te, self.layer_sizes[2], [3, 3], strides=(1, 1), padding='SAME')
                te = tf.reduce_mean(te, axis=[1, 2])

        self.reuse = True
        return te

class TaskContextEncoder:
    def __init__(self,  batch_size, method='mean'):
        """
        :param layer_sizes: A list containing the neuron numbers per layer e.g. [100, 100, 100] returns a 3 layer, 100
                                                                                                        neuron bid-LSTM
                                                                                                        [32]
        :param batch_size: The experiments batch size, useless here
        """
        self.reuse = False
        self.tasktrans = TaskTransformer()
        self.batch_size = batch_size
        self.method = method

    def __call__(self, support_set_embeddings, training=False, name='TaskContext'):
        """
        :param support_set_embeddings: a list of tensor (bs, k*n, w, h, c)
        :param training:
        :param name:
        :return:
        """
        [bs, kn, w, h, c] = support_set_embeddings.get_shape().as_list()
        support_set_embeddings = tf.reshape(support_set_embeddings, shape=[bs*kn, w, h, c])
        # feature transformer
        with tf.variable_scope(name_or_scope=name, reuse=self.reuse):
            if self.method == 'mean':
                t_context = self.tasktrans(support_set_embeddings, training=training)  # (bs*kn, w1,h1,c1)
                t_context = tf.reshape(t_context, shape=[bs, kn, -1])
                t_context = tf.reduce_mean(t_context, axis=1) # (bs, num_features)
                print("t_context shape ", t_context.get_shape()) # (32, 64)
            elif self.method == 'bilstm':
                ## todo add bilstm implementation, previous implementation fails
                pass
            else:
                raise TypeError("No Such Methods, please use mean")
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        print_params(self.variables, name='Task Context Module')
        return t_context

# feature extractor, all meta convolutions
class Classifier:
    def __init__(self, batch_size):
        """
        Fully Convolutional Network using meta convolution
        :param batch_size: Batch size for experiment
        :param layer_sizes: A list of length 4 containing the layer sizes
        :param num_channels: Number of channels of images
        """
        self.reuse = False
        self.batch_size = batch_size
        self.meta_conv = MetaConvolution()
        self.layer_sizes = [64, 64]
        assert len(self.layer_sizes) == 2, "layer_sizes should be a list of length 2"

    def __call__(self, image_embedding, task_context, training=False, keep_prob=1.0):
        """
        Runs the CNN producing the embeddings and the gradients.
        :param image_input: Image input to produce embeddings for.
        :param training: A flag indicating training or evaluation
        :param keep_prob: A tf placeholder of type tf.float32 indicating the amount of dropout applied
        :return: Embeddings of size [batch_size, 64]
        """
        print("task_context shape ", task_context.get_shape())
        with tf.variable_scope('Classifier', reuse=self.reuse):
            # 11*11
            with tf.variable_scope("meta_conv1"):
                m_conv1, m_conv1_w, m_conv1_b = self.meta_conv(image_embedding, task_context, self.layer_sizes[0], [3, 3], training=training)
                m_conv1 = relu(m_conv1, name='outputs')
                m_conv1 = max_pool(m_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME')
            # 6*6
            with tf.variable_scope("meta_conv2"):
                m_conv2, m_conv2_w, m_conv2_b = self.meta_conv(m_conv1, task_context, self.layer_sizes[1], [3, 3], training=training)
                m_conv2 = tf.contrib.layers.flatten(m_conv2)
            print("m_conv2 ", m_conv2.get_shape())
            gen_weights_list = [m_conv1_w, m_conv1_b, m_conv2_w, m_conv2_b]


        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Classifier')
        # print_params(self.variables, name="Feature Extractor")
        return m_conv2, gen_weights_list

# feature extractor
class Extractor:
    def __init__(self):
        """
        Builds a meta CNN to produce embeddings, the final layer weights are generated via meta network
        :param layer_sizes: A list of length 4 containing the layer sizes
        """
        self.reuse = False
        self.layer_sizes = [64, 64, 64, 64]
        assert len(self.layer_sizes) == 4, "layer_sizes should be a list of length 4"

    def __call__(self, support_target_images, training=False, keep_prob=1.0):
        """
        Runs the CNN producing the embeddings and the gradients.
        :param image_input: Image input to produce embeddings for. [batch_size, 28, 28, 1]
        :param training: A flag indicating training or evaluation
        :param keep_prob: A tf placeholder of type tf.float32 indicating the amount of dropout applied
        :return: Embeddings of size [batch_size, 64]
        """
        [bs, kn, w, h, c] = support_target_images.get_shape().as_list()
        support_target_images = tf.reshape(support_target_images, shape=[bs*kn, w, h, c])
        with tf.variable_scope('extractor', reuse=self.reuse):

            with tf.variable_scope('conv_layers'):
                # 84*84
                with tf.variable_scope('g_conv1'):
                    g_conv1_encoder = tf.layers.conv2d(support_target_images, self.layer_sizes[0], [3, 3], strides=(1, 1),
                                                       padding='SAME')
                    g_conv1_encoder = tf.contrib.layers.batch_norm(g_conv1_encoder, updates_collections=None, decay=0.99,
                                                                   scale=True, center=True, is_training=training)

                    g_conv1_encoder = relu(g_conv1_encoder, name='outputs')
                    g_conv1_encoder = max_pool(g_conv1_encoder, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                               padding='SAME')
                    g_conv1_encoder = tf.nn.dropout(g_conv1_encoder, keep_prob=keep_prob)
                # 42*42
                with tf.variable_scope('g_conv2'):
                    g_conv2_encoder = tf.layers.conv2d(g_conv1_encoder, self.layer_sizes[1], [3, 3], strides=(1, 1),
                                                       padding='SAME')
                    g_conv2_encoder = tf.contrib.layers.batch_norm(g_conv2_encoder, updates_collections=None, decay=0.99,
                                                                   scale=True, center=True, is_training=training)

                    g_conv2_encoder = relu(g_conv2_encoder, name='outputs')
                    g_conv2_encoder = max_pool(g_conv2_encoder, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                               padding='SAME')

                # 21*21
                with tf.variable_scope('g_conv3'):
                    g_conv3_encoder = tf.layers.conv2d(g_conv2_encoder, self.layer_sizes[2], [3, 3], strides=(1, 1),
                                                       padding='SAME')
                    g_conv3_encoder = tf.contrib.layers.batch_norm(g_conv3_encoder, updates_collections=None, decay=0.99,
                                                                   scale=True, center=True, is_training=training)

                    g_conv3_encoder = relu(g_conv3_encoder, name='outputs')
                    g_conv3_encoder = max_pool(g_conv3_encoder, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                               padding='SAME')
                # 11*11
                with tf.variable_scope('g_conv4'):
                    g_conv4_encoder = tf.layers.conv2d(g_conv3_encoder, self.layer_sizes[3], [3, 3], strides=(1, 1),
                                                       padding='SAME')
                    g_conv4_encoder = tf.contrib.layers.batch_norm(g_conv4_encoder, updates_collections=None, decay=0.99,
                                                                   scale=True, center=True, is_training=training)
                    g_conv4_encoder = relu(g_conv4_encoder, name='outputs') # ?


        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='conv_layers')
        [bskn, we, he, ce] = g_conv4_encoder.get_shape().as_list()
        embeddings = tf.reshape(g_conv4_encoder, [bs, kn, we, he, ce])

        # print_params(self.variables, name="Feature Extractor")
        return embeddings


class MetaMatchingNetwork:
    def __init__(self, support_set_images, support_set_labels, target_image, target_label, keep_prob,
                 batch_size=32, is_training=False, learning_rate=0.001, rotate_flag=False, num_classes_per_set=5,
                 num_samples_per_class=1, task_method="mean"):

        """
        Builds a matching network, the training and evaluation ops as well as data augmentation routines.
        :param support_set_images: A tensor containing the support set images [batch_size, sequence_size, 28, 28, 1]
        :param support_set_labels: A tensor containing the support set labels [batch_size, sequence_size, 1]
        :param target_image: A tensor containing the target image (image to produce label for) [batch_size, 28, 28, 1]
        :param target_label: A tensor containing the target label [batch_size, 1]
        :param keep_prob: A tf placeholder of type tf.float32 denotes the amount of dropout to be used
        :param batch_size: The batch size for the experiment
        :param num_channels: Number of channels of the images
        :param is_training: Flag indicating whether we are training or evaluating
        :param rotate_flag: Flag indicating whether to rotate the images; This is useless!!!!!!!!!!!!!!
        :param num_classes_per_set: Integer indicating the number of classes per set
        :param num_samples_per_class: Integer indicating the number of samples per class
        :param task_method: Choose from "mean"
        """

        self.batch_size = batch_size
        self.Classifier = Classifier(self.batch_size)
        self.tce = TaskContextEncoder(batch_size=self.batch_size, method=task_method)
        self.dn = DistanceNetwork(metric='cosine')
        self.extractor = Extractor()
        self.classify = AttentionalClassify()
        self.support_set_images = support_set_images
        self.support_set_labels = support_set_labels
        self.target_image = target_image
        self.target_label = target_label

        self.keep_prob = keep_prob
        self.is_training = is_training
        self.rotate_flag = rotate_flag
        self.num_classes_per_set = num_classes_per_set
        self.num_samples_per_class = num_samples_per_class
        self.learning_rate = learning_rate
        self.tensor_list = []

    def loss(self):
        """
        Builds tf graph for Matching Networks, produces losses and summary statistics.
        :return:
        """
        with tf.name_scope("losses"):
            [b, num_classes, spc] = self.support_set_labels.get_shape().as_list()

            self.support_set_labels_ = tf.reshape(self.support_set_labels, shape=(b, num_classes * spc))
            self.support_set_labels_ = tf.one_hot(self.support_set_labels_, self.num_classes_per_set)  # one hot encode

            [b, num_classes, spc, h, w, c] = self.support_set_images.get_shape().as_list()
            self.support_set_images_ = tf.reshape(self.support_set_images, shape=(b,  num_classes*spc, h, w, c))

            ## zero step: extractor feature embeddings
            self.target_image_ = tf.expand_dims(self.target_image, axis=1) #(b, 1, h, w, c)
            ## merge support set and target set, in order to share the feature extractors
            support_target_images = tf.concat([self.support_set_images_, self.target_image_], axis=1) #(b, n*k+1, h, w, c)
            print("+++ support_target images ", support_target_images.get_shape()) # (32, 6, 84, 84, 3)
            print("+++ support_target images [:-1]", support_target_images[:, :-1].get_shape()) # (32, 5, 84, 84, 3)
            support_target_embeddings = self.extractor(support_target_images, training=self.is_training, keep_prob=self.keep_prob)
            print("+++", support_target_embeddings.get_shape()) # (32, 6, 6, 6 , 96)  the last dimension is feature dimension

            ## first  step: generate task feature representation by using support set features
            task_contexts = self.tce(support_target_embeddings[:, :-1], training=self.is_training) # (bs, num_task_features) (32, 64)

            ## second step: transform images via conditional meta task convolution
            trans_support_images_list = []
            trans_target_images_list = []
            tasks_gen_weights_list = [] # todo test generated weights distribution
            for i, (tc, ste) in enumerate(zip(tf.unstack(task_contexts), tf.unstack(support_target_embeddings))):
                print("============ In task instance ", i)
                # support task image embeddings for one task
                steb, gen_weights_list = self.Classifier(image_embedding=ste, task_context=tc, training=self.is_training, keep_prob=self.keep_prob) # (6, 4608)
                trans_support_images_list.append(steb[:-1])
                trans_target_images_list.append(steb[-1])
                tasks_gen_weights_list.append(gen_weights_list)

            trans_support = tf.stack(trans_support_images_list)
            trans_target = tf.stack(trans_target_images_list)
            print("=="*10)  # shape error
            print("trans support set shape and target shape ", trans_support.get_shape(), trans_target.get_shape()) # (32, 5, 4608) (32, 4608)

            similarities = self.dn(support_set=trans_support, input_image=trans_target, name="distance_calculation",
                                   training=self.is_training)  #get similarity between support set embeddings and target

            preds = self.classify(similarities, support_set_y=self.support_set_labels_, name='classify', training=self.is_training)

            if self.batch_size == 1:
                print("If preds is batchsize = 1, reshape it to avoid shape error.")
                preds = tf.reshape(preds, shape=(self.batch_size, preds.get_shape().as_list()[-1]))
            print("preds shape ", preds.get_shape(), tf.argmax(preds, 1).get_shape()) # (bs, num_classes)
            print("target label shape ", self.target_label.get_shape())
            # produce predictions for target probabilities
            correct_prediction = tf.equal(tf.argmax(preds, 1), tf.cast(self.target_label, tf.int64))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            targets = tf.one_hot(self.target_label, self.num_classes_per_set)
            print("targets shape one hot ", targets.get_shape())
            crossentropy_loss = tf.reduce_mean(-tf.reduce_sum(targets * tf.log(preds),
                                                              reduction_indices=[1]))
            print(crossentropy_loss)

            tf.add_to_collection('crossentropy_losses', crossentropy_loss)
            tf.add_to_collection('accuracy', accuracy)

        # todo why return like this, rather than a better string keyworkds?
        return {
            'losses': tf.add_n(tf.get_collection('crossentropy_losses'), name='total_classification_loss'),
            'accuracy': tf.add_n(tf.get_collection('accuracy'), name='accuracy'),
            'preds': preds, # added for ensemble training
            't_label': self.target_label,
            'tasks_gen_weights_list': tasks_gen_weights_list
        }


    def test_ensemble(self, M =1):
        """
        Test using the simpliest ensemble methods: max voting
        But this implemetation is not used, because it is complicated, we just test run the same task instance for
        several times and max voting the results. In experiemnt_builder.py.
        :return:
        """
        with tf.name_scope("losses"):
            [b, num_classes, spc] = self.support_set_labels.get_shape().as_list()
            print("data type ", self.support_set_labels.dtype)
            self.support_set_labels_ = tf.reshape(self.support_set_labels, shape=(b, num_classes * spc))
            print("data type ", self.support_set_labels.dtype)
            self.support_set_labels_ = tf.one_hot(self.support_set_labels_, self.num_classes_per_set)  # one hot encode

            [b, num_classes, spc, h, w, c] = self.support_set_images.get_shape().as_list()
            support_set_images_ = tf.reshape(self.support_set_images, shape=(b,  num_classes*spc, h, w, c))

            ## zero step: extractor feature embeddings
            target_image_ = tf.expand_dims(self.target_image, axis=1) #(b, 1, h, w, c)
            ## merge support set and target set, in order to share the feature extractors
            support_target_images = tf.concat([support_set_images_, target_image_], axis=1) #(b, n*k+1, h, w, c)
            print("+++ support_target images ", support_target_images.get_shape()) # (32, 6, 84, 84, 3)
            print("+++ support_target images [:-1]", support_target_images[:, :-1].get_shape()) # (32, 5, 84, 84, 3)
            support_target_embeddings = self.extractor(support_target_images, training=self.is_training, keep_prob=self.keep_prob)
            print("+++", support_target_embeddings.get_shape()) # (32, 6, 6, 6 , 96)  the last dimension is feature dimension

            ## first  step: generate task feature representation
            task_contexts = self.tce(support_target_embeddings[:, :-1], training=self.is_training) # (bs, num_task_features) (32, 64)

            ## second step: transform images via conditional meta task convolution
            ## todo In order to generate ensemble weights for the same task instance, we just need to run generation network several times
            ensemble_preds = []
            for m in range(M):
                trans_support_images_list = []
                trans_target_images_list = []
                for i, (tc, ste) in enumerate(zip(tf.unstack(task_contexts), tf.unstack(support_target_embeddings))):
                    print("============ In task instance ", i)
                    # support task image embeddings for one task
                    steb = self.Classifier(image_embedding=ste, task_context=tc, training=self.is_training, keep_prob=self.keep_prob)  #(6, 4608)
                    trans_support_images_list.append(steb[:-1])
                    trans_target_images_list.append(steb[-1])


                trans_support = tf.stack(trans_support_images_list)
                trans_target = tf.stack(trans_target_images_list)
                print("==" * 10)  # shape error
                print("trans support set shape and target shape ", trans_support.get_shape(), trans_target.get_shape())

                similarities = self.dn(support_set=trans_support, input_image=trans_target, name="distance_calculation",
                                       training=self.is_training)  # get similarity between support set embeddings and target

                preds = self.classify(similarities,
                                      support_set_y=self.support_set_labels_, name='classify', training=self.is_training)
                print("preds shape ", preds.get_shape())  # (bs, num_classes)
                ensemble_preds.append(tf.arg_max(preds, 1))

            ensemble_preds = tf.stack(ensemble_preds)


        return ensemble_preds


    def train(self, losses):
        """
        Builds the train op
        :param losses: A dictionary containing the losses
        :param learning_rate: Learning rate to be used for Adam
        :param beta1: Beta1 to be used for Adam
        :return:
        """
        c_opt = tf.train.AdamOptimizer(beta1=0.9, learning_rate=self.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # Needed for correct batch norm usage
        with tf.control_dependencies(update_ops):  # Needed for correct batch norm usage
            train_variables = tf.trainable_variables() # all variables
            c_error_opt_op = c_opt.minimize(losses['losses'],
                                            var_list=train_variables)
        print_params(train_variables, "All trainable variables")

        return c_error_opt_op, train_variables


    def init_train(self):
        """
        Get all ops, as well as all losses.
        :return:
        """
        losses = self.loss()
        c_error_opt_op, trainable_variables = self.train(losses)
        summary = tf.summary.merge_all() # summary is not used

        # construct gradient check operation
        check_var_list = trainable_variables
        print_params(check_var_list, "check_var_list")
        grads_list = tf.gradients(losses['losses'], check_var_list)

        # print_params(grads_list, "gradient_list")
        self.grad_var_dict = {'var': check_var_list, 'grad': grads_list}

        return summary, losses, c_error_opt_op


    def check_gradients_magnitude(self, sess, feed_dict):
        """
        Using self.all_trainable_variables and self.losses to compute the gradients of
        :param sess:
        :param feed_dict:
        :return:
        """
        print("check gradients")
        print("name, grad norm, mean, std, max, min | var norm, mean, std, max, min")
        grad_values = sess.run(self.grad_var_dict['grad'], feed_dict=feed_dict)
        var_values = sess.run(self.grad_var_dict['var'], feed_dict=feed_dict)
        for var, g_value, v_value in zip(self.grad_var_dict['var'], grad_values, var_values):
            print(var.name, np.linalg.norm(g_value), np.mean(g_value), np.std(g_value), np.max(g_value), np.min(g_value), "|",
                  np.linalg.norm(v_value), np.mean(v_value), np.std(v_value), np.max(v_value), np.min(v_value))

    def check_tensors_magnitude(self, sess, feed_dict):
        print("check meta convolution weights================")
        tensors = sess.run(self.tensor_list, feed_dict=feed_dict)
        for t, t_v in zip(self.tensor_list, tensors):
            print(t.name, np.linalg.norm(t_v), np.mean(t_v), np.std(t_v), np.max(t_v), np.min(t_v))


    def check_g(self, sess):
        g1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Classifier/meta_conv1/MetaNetwork_meta_conv/normalize_weights/")
        g2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Classifier/meta_conv2/MetaNetwork_meta_conv/normalize_weights/")
        g1_, g2_ = sess.run([g1, g2])
        print("g1 ", g1_)
        print("g2 ", g2_)



