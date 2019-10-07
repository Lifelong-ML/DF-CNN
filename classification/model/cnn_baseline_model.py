import numpy as np
import tensorflow as tf

from utils.utils import *
from utils.utils_nn import *


###############################################################
#### Single task learner (CNN + FC) for Multi-task setting ####
###############################################################
#### Convolutional & Fully-connected Neural Net
class MTL_several_CNN_minibatch():
    def __init__(self, num_tasks, dim_channels, dim_fcs, input_size, dim_kernel, dim_strides, batch_size, learning_rate, learning_rate_decay, padding_type='SAME', max_pooling=False, dim_pool=None, dropout=False, cnn_params=None, fc_params=None):
        self.num_tasks = num_tasks
        #### num_layers == len(channels_size) + len(fc_size) && len(fc_size) >= 1
        self.num_layers = len(dim_channels) + len(dim_fcs)
        self.cnn_channels_size = [input_size[-1]]+dim_channels    ## include dim of input channel

        self.cnn_kernel_size = dim_kernel     ## len(kernel_size) == 2*(len(channels_size)-1)
        self.cnn_stride_size = dim_strides
        self.pool_size = dim_pool      ## len(pool_size) == 2*(len(channels_size)-1)
        self.fc_size = dim_fcs
        self.input_size = input_size    ## img_width * img_height * img_channel
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size

        if cnn_params is None:
            cnn_params = [None for _ in range(self.num_tasks)]
        if fc_params is None:
            fc_params = [None for _ in range(self.num_tasks)]

        #### placeholder of model
        self.epoch = tf.placeholder(dtype=tf.float32)
        self.dropout_prob = tf.placeholder(dtype=tf.float32)

        self.model_input = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_size[0]*self.input_size[1]*self.input_size[2]]) for _ in range(self.num_tasks)]
        self.true_output = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size]) for _ in range(self.num_tasks)]

        with tf.name_scope('Data_Minibatch'):
            train_x_batch = [tf.reshape(x, [-1] + self.input_size) for x in self.model_input]
            valid_x_batch = [tf.reshape(x, [-1] + self.input_size) for x in self.model_input]
            test_x_batch = [tf.reshape(x, [-1] + self.input_size) for x in self.model_input]
            train_y_batch = [y for y in self.true_output]
            valid_y_batch = [y for y in self.true_output]
            test_y_batch = [y for y in self.true_output]

        #### layers of model for train data
        self.train_models = []
        self.param, self.cnn_param, self.fc_param = [], [], []
        for task_cnt in range(self.num_tasks):
            model_tmp, cnn_param_tmp, fc_param_tmp = new_cnn_fc_net(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=cnn_params[task_cnt], fc_params=fc_params[task_cnt], padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, output_type=None)
            self.train_models.append(model_tmp)
            self.cnn_param.append(cnn_param_tmp)
            self.fc_param.append(fc_param_tmp)
            self.param = self.param + cnn_param_tmp + fc_param_tmp

        #### layers of model for validation data
        self.valid_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _ = new_cnn_fc_net(valid_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=self.cnn_param[task_cnt], fc_params=self.fc_param[task_cnt], padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, output_type=None)
            self.valid_models.append(model_tmp)

        #### layers of model for test data
        self.test_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _ = new_cnn_fc_net(test_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=self.cnn_param[task_cnt], fc_params=self.fc_param[task_cnt], padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, output_type=None)
            self.test_models.append(model_tmp)

        #### functions of model
        self.train_eval, self.valid_eval, self.test_eval, self.train_loss, self.valid_loss, self.test_loss, self.train_accuracy, self.valid_accuracy, self.test_accuracy, self.train_output_label, self.valid_output_label, self.test_output_label = mtl_model_output_functions([self.train_models, self.valid_models, self.test_models], [train_y_batch, valid_y_batch, test_y_batch], self.num_tasks)

        self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.train_loss[x]) for x in range(self.num_tasks)]
        self.gradient = [tf.gradients(self.train_loss[x], self.cnn_param[x]+self.fc_param[x]) for x in range(self.num_tasks)]

        self.num_trainable_var = count_trainable_var()

    def get_params_val(self, sess):
        cnn_param_val = [get_value_of_valid_tensors(sess, cnn_param) for cnn_param in self.cnn_param]
        fc_param_val = [get_value_of_valid_tensors(sess, fc_param) for fc_param in self.fc_param]

        parameters_val = {}
        parameters_val['conv_trainable_weights'] = savemat_wrapper_nested_list(cnn_param_val)
        parameters_val['fc_weights'] = savemat_wrapper_nested_list(fc_param_val)
        return parameters_val


########################################################
####     Single CNN + FC for Multi-task Learning    ####
########################################################
#### Convolutional & Fully-connected Neural Net
class MTL_CNN_minibatch():
    def __init__(self, num_tasks, dim_channels, dim_fcs, input_size, dim_kernel, dim_strides, batch_size, learning_rate, learning_rate_decay, padding_type='SAME', max_pooling=False, dim_pool=None, dropout=False):
        self.num_tasks = num_tasks
        #### num_layers == len(channels_size) + len(fc_size) && len(fc_size) >= 1
        self.num_layers = len(dim_channels) + len(dim_fcs)
        self.cnn_channels_size = [input_size[-1]]+dim_channels    ## include dim of input channel
        self.cnn_kernel_size = dim_kernel     ## len(kernel_size) == 2*(len(channels_size)-1)
        self.cnn_stride_size = dim_strides
        self.pool_size = dim_pool      ## len(pool_size) == 2*(len(channels_size)-1)
        self.fc_size = dim_fcs
        self.input_size = input_size    ## img_width * img_height * img_channel
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size

        #### placeholder of model
        self.epoch = tf.placeholder(dtype=tf.float32)
        self.dropout_prob = tf.placeholder(dtype=tf.float32)

        self.model_input = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_size[0]*self.input_size[1]*self.input_size[2]]) for _ in range(self.num_tasks)]
        self.true_output = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size]) for _ in range(self.num_tasks)]

        with tf.name_scope('Data_Minibatch'):
            train_x_batch = [tf.reshape(x, [-1] + self.input_size) for x in self.model_input]
            valid_x_batch = [tf.reshape(x, [-1] + self.input_size) for x in self.model_input]
            test_x_batch = [tf.reshape(x, [-1] + self.input_size) for x in self.model_input]
            train_y_batch = [y for y in self.true_output]
            valid_y_batch = [y for y in self.true_output]
            test_y_batch = [y for y in self.true_output]

        #### layers of model for train data
        self.train_models = []
        for task_cnt in range(self.num_tasks):
            if task_cnt == 0:
                model_tmp, self.cnn_param, self.fc_param = new_cnn_fc_net(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=None, fc_params=None, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, output_type=None)
            else:
                model_tmp, _, _ = new_cnn_fc_net(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=self.cnn_param, fc_params=self.fc_param, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, output_type=None)
            self.train_models.append(model_tmp)
        self.param = self.cnn_param + self.fc_param

        #### layers of model for validation data
        self.valid_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _ = new_cnn_fc_net(valid_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=self.cnn_param, fc_params=self.fc_param, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, output_type=None)
            self.valid_models.append(model_tmp)

        #### layers of model for test data
        self.test_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _ = new_cnn_fc_net(test_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=self.cnn_param, fc_params=self.fc_param, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, output_type=None)
            self.test_models.append(model_tmp)

        #### functions of model
        self.train_eval, self.valid_eval, self.test_eval, self.train_loss, self.valid_loss, self.test_loss, self.train_accuracy, self.valid_accuracy, self.test_accuracy, self.train_output_label, self.valid_output_label, self.test_output_label = mtl_model_output_functions([self.train_models, self.valid_models, self.test_models], [train_y_batch, valid_y_batch, test_y_batch], self.num_tasks)

        self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.train_loss[x]) for x in range(self.num_tasks)]

        self.num_trainable_var = count_trainable_var()

    def get_params_val(self, sess):
        cnn_param_val = get_value_of_valid_tensors(sess, self.cnn_param)
        fc_param_val = get_value_of_valid_tensors(sess, self.fc_param)

        parameters_val = {}
        parameters_val['conv_trainable_weights'] = savemat_wrapper(cnn_param_val)
        parameters_val['fc_weights'] = savemat_wrapper(fc_param_val)
        return parameters_val


########################################################
####      Single CNN + Task-specific FC for MTL     ####
########         Hard Parameter Sharing         ########
########################################################
#### Convolutional & Fully-connected Neural Net
class MTL_CNN_HPS_minibatch():
    def __init__(self, num_tasks, dim_channels, dim_fcs, input_size, dim_kernel, dim_strides, batch_size, learning_rate, learning_rate_decay, padding_type='SAME', max_pooling=False, dim_pool=None, dropout=False):
        self.num_tasks = num_tasks
        #self.num_layers = [len(dim_fcs[0])] + [len(dim_fcs[1][x]) for x in range(self.num_tasks)]
        self.cnn_channels_size = [input_size[-1]]+dim_channels    ## include dim of input channel
        self.cnn_kernel_size = dim_kernel     ## len(kernel_size) == 2*(len(channels_size)-1)
        self.cnn_stride_size = dim_strides
        self.num_TS_fc_param = 0
        self.pool_size = dim_pool      ## len(pool_size) == 2*(len(channels_size)-1)
        self.fc_size = dim_fcs
        self.input_size = input_size    ## img_width * img_height * img_channel
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size

        #### placeholder of model
        self.epoch = tf.placeholder(dtype=tf.float32)
        self.dropout_prob = tf.placeholder(dtype=tf.float32)

        self.model_input = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_size[0]*self.input_size[1]*self.input_size[2]]) for _ in range(self.num_tasks)]
        self.true_output = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size]) for _ in range(self.num_tasks)]

        with tf.name_scope('Data_Minibatch'):
            train_x_batch = [tf.reshape(x, [-1] + self.input_size) for x in self.model_input]
            valid_x_batch = [tf.reshape(x, [-1] + self.input_size) for x in self.model_input]
            test_x_batch = [tf.reshape(x, [-1] + self.input_size) for x in self.model_input]
            train_y_batch = [y for y in self.true_output]
            valid_y_batch = [y for y in self.true_output]
            test_y_batch = [y for y in self.true_output]

        #### layers of model for train data
        self.train_models, self.cnn_param, self.fc_param = new_hardparam_cnn_fc_nets(train_x_batch, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.num_tasks, cnn_params=None, fc_params=None, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, output_type=None)
        self.param = self.cnn_param + sum(self.fc_param, [])

        #### layers of model for validation data
        self.valid_models, _, _ = new_hardparam_cnn_fc_nets(valid_x_batch, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.num_tasks, cnn_params=self.cnn_param, fc_params=self.fc_param, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, output_type=None)

        #### layers of model for test data
        self.test_models, _, _ = new_hardparam_cnn_fc_nets(test_x_batch, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.num_tasks, cnn_params=self.cnn_param, fc_params=self.fc_param, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, output_type=None)

        #### functions of model
        self.train_eval, self.valid_eval, self.test_eval, self.train_loss, self.valid_loss, self.test_loss, self.train_accuracy, self.valid_accuracy, self.test_accuracy, self.train_output_label, self.valid_output_label, self.test_output_label = mtl_model_output_functions([self.train_models, self.valid_models, self.test_models], [train_y_batch, valid_y_batch, test_y_batch], self.num_tasks)

        self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.train_loss[x]) for x in range(self.num_tasks)]

        self.num_trainable_var = count_trainable_var()

    def get_params_val(self, sess):
        cnn_param_val = get_value_of_valid_tensors(sess, self.cnn_param)
        fc_param_val = [get_value_of_valid_tensors(sess, fc_param) for fc_param in self.fc_param]

        parameters_val = {}
        parameters_val['conv_trainable_weights'] = savemat_wrapper(cnn_param_val)
        parameters_val['fc_weights'] = savemat_wrapper_nested_list(fc_param_val)
        return parameters_val


########################################################
####     CNN + FC model for Multi-task Learning     ####
########          Tensor Factorization          ########
########################################################
#### Convolutional & Fully-connected Neural Net
class MTL_CNN_tensorfactor_minibatch():
    def __init__(self, num_tasks, dim_channels, dim_fcs, input_size, dim_kernel, dim_strides, batch_size, learning_rate, learning_rate_decay, padding_type='SAME', max_pooling=False, dim_pool=None, dropout=False, factor_type='Tucker', factor_eps_or_k=0.01, init_param=None):
        self.num_tasks = num_tasks
        self.cnn_channels_size = [input_size[-1]]+dim_channels    ## include dim of input channel
        self.cnn_kernel_size = dim_kernel     ## len(kernel_size) == 2*(len(channels_size)-1)
        self.cnn_stride_size = dim_strides
        self.pool_size = dim_pool      ## len(pool_size) == 2*(len(channels_size)-1)
        self.fc_size = dim_fcs
        self.input_size = input_size    ## img_width * img_height * img_channel
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size

        #### placeholder of model
        self.epoch = tf.placeholder(dtype=tf.float32)
        self.dropout_prob = tf.placeholder(dtype=tf.float32)

        self.model_input = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_size[0]*self.input_size[1]*self.input_size[2]]) for _ in range(self.num_tasks)]
        self.true_output = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size]) for _ in range(self.num_tasks)]

        with tf.name_scope('Data_Minibatch'):
            train_x_batch = [tf.reshape(x, [-1] + self.input_size) for x in self.model_input]
            valid_x_batch = [tf.reshape(x, [-1] + self.input_size) for x in self.model_input]
            test_x_batch = [tf.reshape(x, [-1] + self.input_size) for x in self.model_input]
            train_y_batch = [y for y in self.true_output]
            valid_y_batch = [y for y in self.true_output]
            test_y_batch = [y for y in self.true_output]

        #### layers of model for train data
        self.train_models, self.cnn_param, self.fc_param = new_tensorfactored_cnn_fc_nets(train_x_batch, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.num_tasks, cnn_params=None, fc_params=None, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, output_type=None, factor_type=factor_type, factor_eps_or_k=factor_eps_or_k, init_param=init_param)
        self.param = self.cnn_param + self.fc_param

        #### layers of model for validation data
        self.valid_models, _, _ = new_tensorfactored_cnn_fc_nets(valid_x_batch, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.num_tasks, cnn_params=self.cnn_param, fc_params=self.fc_param, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, output_type=None, factor_type=factor_type, factor_eps_or_k=factor_eps_or_k)

        #### layers of model for test data
        self.test_models, _, _ = new_tensorfactored_cnn_fc_nets(test_x_batch, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.num_tasks, cnn_params=self.cnn_param, fc_params=self.fc_param, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, output_type=None, factor_type=factor_type, factor_eps_or_k=factor_eps_or_k)

        #### functions of model
        self.train_eval, self.valid_eval, self.test_eval, self.train_loss, self.valid_loss, self.test_loss, self.train_accuracy, self.valid_accuracy, self.test_accuracy, self.train_output_label, self.valid_output_label, self.test_output_label = mtl_model_output_functions([self.train_models, self.valid_models, self.test_models], [train_y_batch, valid_y_batch, test_y_batch], self.num_tasks)

        self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.train_loss[x]) for x in range(self.num_tasks)]

        self.num_trainable_var = count_trainable_var()

    def get_params_val(self, sess):
        cnn_param_val = get_value_of_valid_tensors(sess, self.cnn_param)
        fc_param_val = get_value_of_valid_tensors(sess, self.fc_param)

        parameters_val = {}
        parameters_val['conv_trainable_weights'] = savemat_wrapper(cnn_param_val)
        parameters_val['fc_weights'] = savemat_wrapper(fc_param_val)
        return parameters_val


########################################################
####     CNN + FC model for Multi-task Learning     ####
########         Progressive Neural Net         ########
########################################################
#### Convolutional & Fully-connected Neural Net
class MTL_CNN_progressive_minibatch():
    def __init__(self, dim_channels, dim_fcs, input_size, dim_kernel, dim_strides, batch_size, learning_rate, learning_rate_decay, padding_type='SAME', max_pooling=False, dim_pool=None, dropout=False, dim_reduction_scale=1.0):
        self.num_tasks = 1
        self.cnn_channels_size = [input_size[-1]]+dim_channels    ## include dim of input channel
        self.cnn_kernel_size = dim_kernel     ## len(kernel_size) == 2*(len(channels_size)-1)
        self.cnn_stride_size = dim_strides
        self.pool_size = dim_pool      ## len(pool_size) == 2*(len(channels_size)-1)
        self.fc_size = dim_fcs
        self.input_size = input_size    ## img_width * img_height * img_channel
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size

        self.padding_type = padding_type
        self.max_pooling = max_pooling
        self.dropout = dropout
        self.dim_reduction_scale=dim_reduction_scale

        self.num_trainable_var = 0
        self.regenerate_network(prev_net_param=None)

    def regenerate_network(self, prev_net_param=None):
        # prev_net_param : [ [cnn_0, fc_0, lat_param_0], [cnn_1, fc_1, lat_param_1], ..., [net_param for num_tasks-2] ]
        assert ((prev_net_param is None and self.num_tasks == 1) or (prev_net_param is not None and self.num_tasks > 1)), "Parameters of previous columns are in wrong format"

        #### placeholder of model
        self.epoch = tf.placeholder(dtype=tf.float32)
        self.dropout_prob = tf.placeholder(dtype=tf.float32)

        self.model_input = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_size[0]*self.input_size[1]*self.input_size[2]])
        self.true_output = tf.placeholder(dtype=tf.float32, shape=[self.batch_size])
        self.model_input_batch = tf.reshape(self.model_input, [-1] + self.input_size)
        # model_input_batch instead of x_batch // true_output instead of y_batch

        #### layers of model for train data
        self.param, self.non_trainable_param = [], []
        self.train_models, self.cnn_params, self.cnn_lateral_params, self.fc_params = [], [], [], []
        # model with pre-trained and fixed param
        for task_cnt in range(self.num_tasks-1):
            # inside here, prev_net_param is not None
            if task_cnt < 1:
                model_tmp, cnn_param_tmp, cnn_lat_param_tmp, fc_param_tmp = new_progressive_cnn_fc_net(self.model_input_batch, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=prev_net_param[task_cnt][0], fc_params=prev_net_param[task_cnt][1], padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, output_type=None, prev_net=None, cnn_lateral_params=None, trainable=False, dim_reduction_scale=self.dim_reduction_scale, use_numpy_var_in_graph=True)
            else:
                model_tmp, cnn_param_tmp, cnn_lat_param_tmp, fc_param_tmp = new_progressive_cnn_fc_net(self.model_input_batch, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=prev_net_param[task_cnt][0], fc_params=prev_net_param[task_cnt][1], padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, output_type=None, prev_net=self.train_models, cnn_lateral_params=prev_net_param[task_cnt][2], trainable=False, dim_reduction_scale=self.dim_reduction_scale, use_numpy_var_in_graph=True)
            self.train_models.append(model_tmp)
            self.cnn_params.append(cnn_param_tmp)
            self.cnn_lateral_params.append(cnn_lat_param_tmp)
            self.fc_params.append(fc_param_tmp)
            self.non_trainable_param = self.non_trainable_param + [para if para is not None else [] for para in cnn_param_tmp] + [para if para is not None else [] for para in fc_param_tmp] + [para if para is not None else [] for para in cnn_lat_param_tmp]

        # model with trainable param
        if self.num_tasks < 2:
            model_tmp, cnn_param_tmp, cnn_lat_param_tmp, fc_param_tmp = new_progressive_cnn_fc_net(self.model_input_batch, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=None, fc_params=None, padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, output_type=None, prev_net=None, cnn_lateral_params=None, trainable=True, dim_reduction_scale=self.dim_reduction_scale)
        else:
            model_tmp, cnn_param_tmp, cnn_lat_param_tmp, fc_param_tmp = new_progressive_cnn_fc_net(self.model_input_batch, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=None, fc_params=None, padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, output_type=None, prev_net=self.train_models, cnn_lateral_params=None, trainable=True, dim_reduction_scale=self.dim_reduction_scale)
        self.train_models.append(model_tmp)
        self.cnn_params.append(cnn_param_tmp)
        self.cnn_lateral_params.append(cnn_lat_param_tmp)
        self.fc_params.append(fc_param_tmp)
        self.param = self.param + [para if para is not None else [] for para in cnn_param_tmp] + [para if para is not None else [] for para in fc_param_tmp] + [para if para is not None else [] for para in cnn_lat_param_tmp]

        #### layers of model for validation data
        self.valid_models = []
        for task_cnt in range(self.num_tasks):
            if task_cnt < 1:
                model_tmp, _, _, _ = new_progressive_cnn_fc_net(self.model_input_batch, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=self.cnn_params[task_cnt], fc_params=self.fc_params[task_cnt], padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, output_type=None, prev_net=None, cnn_lateral_params=self.cnn_lateral_params[task_cnt], trainable=False, dim_reduction_scale=self.dim_reduction_scale, use_numpy_var_in_graph=True)
            else:
                model_tmp, _, _, _ = new_progressive_cnn_fc_net(self.model_input_batch, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=self.cnn_params[task_cnt], fc_params=self.fc_params[task_cnt], padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, output_type=None, prev_net=self.valid_models, cnn_lateral_params=self.cnn_lateral_params[task_cnt], trainable=False, dim_reduction_scale=self.dim_reduction_scale, use_numpy_var_in_graph=(task_cnt<self.num_tasks-1))
            self.valid_models.append(model_tmp)

        #### layers of model for test data
        self.test_models = []
        for task_cnt in range(self.num_tasks):
            if task_cnt < 1:
                model_tmp, _, _, _ = new_progressive_cnn_fc_net(self.model_input_batch, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=self.cnn_params[task_cnt], fc_params=self.fc_params[task_cnt], padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, output_type=None, prev_net=None, cnn_lateral_params=self.cnn_lateral_params[task_cnt], trainable=False, dim_reduction_scale=self.dim_reduction_scale, use_numpy_var_in_graph=True)
            else:
                model_tmp, _, _, _ = new_progressive_cnn_fc_net(self.model_input_batch, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=self.cnn_params[task_cnt], fc_params=self.fc_params[task_cnt], padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, output_type=None, prev_net=self.test_models, cnn_lateral_params=self.cnn_lateral_params[task_cnt], trainable=False, dim_reduction_scale=self.dim_reduction_scale, use_numpy_var_in_graph=(task_cnt<self.num_tasks-1))
            self.test_models.append(model_tmp)

        #### functions of model
        self.train_eval, self.valid_eval, self.test_eval, self.train_loss, self.valid_loss, self.test_loss, self.train_accuracy, self.valid_accuracy, self.test_accuracy, self.train_output_label, self.valid_output_label, self.test_output_label = mtl_model_output_functions([self.train_models, self.valid_models, self.test_models], [[self.true_output for _ in range(self.num_tasks)], [self.true_output for _ in range(self.num_tasks)], [self.true_output for _ in range(self.num_tasks)]], self.num_tasks)

        self.update = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate / (1.0 + self.epoch * self.learn_rate_decay)).minimize(self.train_loss[-1])

        ### (Caution) No way to remove existing older computational graph, so TF stores all older networks
        self.num_trainable_var = self.num_trainable_var + count_trainable_var()

    def get_prev_net_param(self, sess):
        # parameter of 0 ~ num_tasks-2 column : numpy array
        # parameter of num_tasks-1 column : tf variable
        prev_trained_net_param = []
        for task_cnt in range(self.num_tasks-1):
            prev_trained_net_param.append([self.cnn_params[task_cnt], self.fc_params[task_cnt], self.cnn_lateral_params[task_cnt]])

        prev_trained_net_param.append([[sess.run(x) for x in self.cnn_params[self.num_tasks-1]], [sess.run(x) for x in self.fc_params[self.num_tasks-1]], [(x if (x is None) else sess.run(x)) for x in self.cnn_lateral_params[self.num_tasks-1]]])
        return prev_trained_net_param

    def new_lifelong_task(self, sess=None, params=None):
        self.num_tasks = self.num_tasks + 1
        if self.num_tasks > 1 and params is None:
            # get pre-trained model param
            assert (sess is not None), "Unable to get actual value of parameter of the model, give correct tf session"
            # prev_net_param : [ [cnn_0, fc_0, lat_param_0], [cnn_1, fc_1, lat_param_1], ..., [net_param for num_tasks-2] ]
            prev_trained_net_param = []
            for task_cnt in range(self.num_tasks-2):
                # they are already list of numpy tensors
                prev_trained_net_param.append([self.cnn_params[task_cnt], self.fc_params[task_cnt], self.cnn_lateral_params[task_cnt]])

            prev_trained_net_param.append([[sess.run(x) for x in self.cnn_params[self.num_tasks-2]], [sess.run(x) for x in self.fc_params[self.num_tasks-2]], [(x if (x is None) else sess.run(x)) for x in self.cnn_lateral_params[self.num_tasks-2]]])
        elif params is not None:
            prev_trained_net_param = params
        else:
            prev_trained_net_param = None
        self.regenerate_network(prev_net_param=prev_trained_net_param)
