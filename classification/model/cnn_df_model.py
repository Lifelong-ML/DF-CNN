import numpy as np
import tensorflow as tf

from utils.utils import *
from utils.utils_df_nn import *


#### Deconvolutional factorized CNN in paper 'Learning shared knowledge for deep lifelong learning using deconvolutional networks' of IJCAI 2019
class Deconvolutional_Factorized_CNN():
    def __init__(self, num_tasks, dim_channels, dim_fcs, input_size, dim_kernels, dim_strides, dim_cnn_know_base, dim_cnn_task_specific, dim_cnn_deconv_strides, batch_size, learning_rate, learning_rate_decay, padding_type='SAME', max_pooling=False, dim_pool=None, dropout=False, relation_activation_fn_cnn=tf.nn.relu):
        self.num_tasks = num_tasks
        self.num_layers = len(dim_channels) + len(dim_fcs)
        self.num_cnn_KB_para = len(dim_channels)
        self.num_cnn_TS_para = 0
        self.num_fc_para = 0
        self.cnn_channels_size = [input_size[-1]]+dim_channels    ## include dim of input channel
        self.cnn_kernel_size = dim_kernels     ## len(kernel_size) == 2*(len(channels_size)-1)
        self.cnn_stride_size = dim_strides
        self.pool_size = dim_pool      ## len(pool_size) == 2*(len(channels_size)-1)
        self.cnn_KB_size = dim_cnn_know_base
        self.cnn_TS_size = dim_cnn_task_specific
        self.cnn_deconv_stride_size = dim_cnn_deconv_strides
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
        self.train_models, self.cnn_TS_param, self.cnn_gen_param, self.fc_param = [], [], [], []
        for task_cnt in range(self.num_tasks):
            if task_cnt == 0:
                model_tmp, self.cnn_KB_param, cnn_TS_param_tmp, cnn_gen_param_tmp, fc_param_tmp = new_dfcnn_fc_net(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=None, cnn_TS_params=None, fc_params=None, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, output_type=None, task_index=task_cnt)
                self.num_cnn_TS_para = len(cnn_TS_param_tmp)
                self.num_fc_para = len(fc_param_tmp)
            else:
                model_tmp, _, cnn_TS_param_tmp, cnn_gen_param_tmp, fc_param_tmp = new_dfcnn_fc_net(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=self.cnn_KB_param, cnn_TS_params=None, fc_params=None, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, output_type=None, task_index=task_cnt)
            self.cnn_TS_param.append(cnn_TS_param_tmp)
            self.cnn_gen_param.append(cnn_gen_param_tmp)
            self.fc_param.append(fc_param_tmp)
            self.train_models.append(model_tmp)
        self.param = self.cnn_KB_param + sum(self.cnn_TS_param, []) + sum(self.fc_param, [])

        assert (len(self.cnn_KB_param) == self.num_cnn_KB_para), "CNN KB size doesn't match"
        assert (sum([len(self.cnn_TS_param[x]) for x in range(len(self.cnn_TS_param))]) == self.num_tasks * self.num_cnn_TS_para), "CNN TS size doesn't match"

        #### layers of model for validation data
        self.valid_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _, _, _ = new_dfcnn_fc_net(valid_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=self.cnn_KB_param, cnn_TS_params=self.cnn_TS_param[task_cnt], fc_params=self.fc_param[task_cnt], padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, output_type=None, task_index=task_cnt)
            self.valid_models.append(model_tmp)

        #### layers of model for test data
        self.test_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _, _, _ = new_dfcnn_fc_net(test_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=self.cnn_KB_param, cnn_TS_params=self.cnn_TS_param[task_cnt], fc_params=self.fc_param[task_cnt], padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, output_type=None, task_index=task_cnt)
            self.test_models.append(model_tmp)

        #### loss functions of model
        self.train_eval, self.valid_eval, self.test_eval, self.train_loss, self.valid_loss, self.test_loss, self.train_accuracy, self.valid_accuracy, self.test_accuracy, self.train_output_label, self.valid_output_label, self.test_output_label = mtl_model_output_functions([self.train_models, self.valid_models, self.test_models], [train_y_batch, valid_y_batch, test_y_batch], self.num_tasks)

        self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.train_loss[x]) for x in range(self.num_tasks)]

        self.num_trainable_var = count_trainable_var()

    def get_params_val(self, sess):
        KB_param_val = get_value_of_valid_tensors(sess, self.cnn_KB_param)
        TS_param_val = [get_value_of_valid_tensors(sess, cnn_TS_param) for cnn_TS_param in self.cnn_TS_param]
        gen_param_val = [get_value_of_valid_tensors(sess, cnn_gen_param) for cnn_gen_param in self.cnn_gen_param]
        fc_param_val = [get_value_of_valid_tensors(sess, fc_param) for fc_param in self.fc_param]

        parameters_val = {}
        parameters_val['conv_KB'] = savemat_wrapper(KB_param_val)
        parameters_val['conv_TS'] = savemat_wrapper_nested_list(TS_param_val)
        parameters_val['conv_generated_weights'] = savemat_wrapper_nested_list(gen_param_val)
        parameters_val['fc_weights'] = savemat_wrapper_nested_list(fc_param_val)
        return parameters_val



#### Ablated architectures of deconvolutional factorized CNN
class Deconvolutional_Factorized_CNN_Direct():
    def __init__(self, num_tasks, dim_channels, dim_fcs, input_size, dim_kernels, dim_strides, dim_cnn_know_base, dim_cnn_task_specific, dim_cnn_deconv_strides, batch_size, learning_rate, learning_rate_decay, padding_type='SAME', max_pooling=False, dim_pool=None, dropout=False, relation_activation_fn_cnn=tf.nn.relu):
        self.num_tasks = num_tasks
        self.num_layers = len(dim_channels) + len(dim_fcs)
        self.num_cnn_KB_para = len(dim_channels)
        self.num_cnn_TS_para = 0
        self.num_fc_para = 0
        self.cnn_channels_size = [input_size[-1]]+dim_channels    ## include dim of input channel
        self.cnn_kernel_size = dim_kernels     ## len(kernel_size) == 2*(len(channels_size)-1)
        self.cnn_stride_size = dim_strides
        self.pool_size = dim_pool      ## len(pool_size) == 2*(len(channels_size)-1)
        self.cnn_KB_size = dim_cnn_know_base
        self.cnn_TS_size = dim_cnn_task_specific
        self.cnn_deconv_stride_size = dim_cnn_deconv_strides
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
        self.train_models, self.cnn_TS_param, self.cnn_gen_param, self.fc_param = [], [], [], []
        for task_cnt in range(self.num_tasks):
            if task_cnt == 0:
                model_tmp, self.cnn_KB_param, cnn_TS_param_tmp, cnn_gen_param_tmp, fc_param_tmp = new_dfcnn_direct_fc_net(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=None, cnn_TS_params=None, fc_params=None, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, output_type=None, task_index=task_cnt)
                self.num_cnn_TS_para = len(cnn_TS_param_tmp)
                self.num_fc_para = len(fc_param_tmp)
            else:
                model_tmp, _, cnn_TS_param_tmp, cnn_gen_param_tmp, fc_param_tmp = new_dfcnn_direct_fc_net(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=self.cnn_KB_param, cnn_TS_params=None, fc_params=None, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, output_type=None, task_index=task_cnt)
            self.cnn_TS_param.append(cnn_TS_param_tmp)
            self.cnn_gen_param.append(cnn_gen_param_tmp)
            self.fc_param.append(fc_param_tmp)
            self.train_models.append(model_tmp)
        self.param = self.cnn_KB_param + sum(self.cnn_TS_param, []) + sum(self.fc_param, [])

        assert (len(self.cnn_KB_param) == self.num_cnn_KB_para), "CNN KB size doesn't match"
        assert (sum([len(self.cnn_TS_param[x]) for x in range(len(self.cnn_TS_param))]) == self.num_tasks * self.num_cnn_TS_para), "CNN TS size doesn't match"

        #### layers of model for validation data
        self.valid_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _, _, _ = new_dfcnn_direct_fc_net(valid_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=self.cnn_KB_param, cnn_TS_params=self.cnn_TS_param[task_cnt], fc_params=self.fc_param[task_cnt], padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, output_type=None, task_index=task_cnt)
            self.valid_models.append(model_tmp)

        #### layers of model for test data
        self.test_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _, _, _ = new_dfcnn_direct_fc_net(test_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=self.cnn_KB_param, cnn_TS_params=self.cnn_TS_param[task_cnt], fc_params=self.fc_param[task_cnt], padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, output_type=None, task_index=task_cnt)
            self.test_models.append(model_tmp)

        #### loss functions of model
        self.train_eval, self.valid_eval, self.test_eval, self.train_loss, self.valid_loss, self.test_loss, self.train_accuracy, self.valid_accuracy, self.test_accuracy, self.train_output_label, self.valid_output_label, self.test_output_label = mtl_model_output_functions([self.train_models, self.valid_models, self.test_models], [train_y_batch, valid_y_batch, test_y_batch], self.num_tasks)

        self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.train_loss[x]) for x in range(self.num_tasks)]

        self.num_trainable_var = count_trainable_var()

    def get_params_val(self, sess):
        KB_param_val = get_value_of_valid_tensors(sess, self.cnn_KB_param)
        TS_param_val = [get_value_of_valid_tensors(sess, cnn_TS_param) for cnn_TS_param in self.cnn_TS_param]
        gen_param_val = [get_value_of_valid_tensors(sess, cnn_gen_param) for cnn_gen_param in self.cnn_gen_param]
        fc_param_val = [get_value_of_valid_tensors(sess, fc_param) for fc_param in self.fc_param]

        parameters_val = {}
        parameters_val['conv_KB'] = savemat_wrapper(KB_param_val)
        parameters_val['conv_TS'] = savemat_wrapper_nested_list(TS_param_val)
        parameters_val['conv_generated_weights'] = savemat_wrapper_nested_list(gen_param_val)
        parameters_val['fc_weights'] = savemat_wrapper_nested_list(fc_param_val)
        return parameters_val


#### Ablated architectures of deconvolutional factorized CNN
class Deconvolutional_Factorized_CNN_tc2():
    def __init__(self, num_tasks, dim_channels, dim_fcs, input_size, dim_kernels, dim_strides, dim_cnn_know_base, dim_cnn_task_specific, dim_cnn_deconv_strides, batch_size, learning_rate, learning_rate_decay, padding_type='SAME', max_pooling=False, dim_pool=None, dropout=False, relation_activation_fn_cnn=tf.nn.relu):
        self.num_tasks = num_tasks
        self.num_layers = len(dim_channels) + len(dim_fcs)
        self.num_cnn_KB_para = len(dim_channels)
        self.num_cnn_TS_para = 0
        self.num_fc_para = 0
        self.cnn_channels_size = [input_size[-1]]+dim_channels    ## include dim of input channel
        self.cnn_kernel_size = dim_kernels     ## len(kernel_size) == 2*(len(channels_size)-1)
        self.cnn_stride_size = dim_strides
        self.pool_size = dim_pool      ## len(pool_size) == 2*(len(channels_size)-1)
        self.cnn_KB_size = dim_cnn_know_base
        self.cnn_TS_size = dim_cnn_task_specific
        self.cnn_deconv_stride_size = dim_cnn_deconv_strides
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
        self.train_models, self.cnn_TS_param, self.cnn_gen_param, self.fc_param = [], [], [], []
        for task_cnt in range(self.num_tasks):
            if task_cnt == 0:
                model_tmp, self.cnn_KB_param, cnn_TS_param_tmp, cnn_gen_param_tmp, fc_param_tmp = new_dfcnn_tc2_fc_net(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=None, cnn_TS_params=None, fc_params=None, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, output_type=None, task_index=task_cnt)
                self.num_cnn_TS_para = len(cnn_TS_param_tmp)
                self.num_fc_para = len(fc_param_tmp)
            else:
                model_tmp, _, cnn_TS_param_tmp, cnn_gen_param_tmp, fc_param_tmp = new_dfcnn_tc2_fc_net(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=self.cnn_KB_param, cnn_TS_params=None, fc_params=None, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, output_type=None, task_index=task_cnt)
            self.cnn_TS_param.append(cnn_TS_param_tmp)
            self.cnn_gen_param.append(cnn_gen_param_tmp)
            self.fc_param.append(fc_param_tmp)
            self.train_models.append(model_tmp)
        self.param = self.cnn_KB_param + sum(self.cnn_TS_param, []) + sum(self.fc_param, [])

        assert (len(self.cnn_KB_param) == self.num_cnn_KB_para), "CNN KB size doesn't match"
        assert (sum([len(self.cnn_TS_param[x]) for x in range(len(self.cnn_TS_param))]) == self.num_tasks * self.num_cnn_TS_para), "CNN TS size doesn't match"

        #### layers of model for validation data
        self.valid_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _, _, _ = new_dfcnn_tc2_fc_net(valid_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=self.cnn_KB_param, cnn_TS_params=self.cnn_TS_param[task_cnt], fc_params=self.fc_param[task_cnt], padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, output_type=None, task_index=task_cnt)
            self.valid_models.append(model_tmp)

        #### layers of model for test data
        self.test_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _, _, _ = new_dfcnn_tc2_fc_net(test_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=self.cnn_KB_param, cnn_TS_params=self.cnn_TS_param[task_cnt], fc_params=self.fc_param[task_cnt], padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, output_type=None, task_index=task_cnt)
            self.test_models.append(model_tmp)

        #### loss functions of model
        self.train_eval, self.valid_eval, self.test_eval, self.train_loss, self.valid_loss, self.test_loss, self.train_accuracy, self.valid_accuracy, self.test_accuracy, self.train_output_label, self.valid_output_label, self.test_output_label = mtl_model_output_functions([self.train_models, self.valid_models, self.test_models], [train_y_batch, valid_y_batch, test_y_batch], self.num_tasks)

        self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.train_loss[x]) for x in range(self.num_tasks)]

        self.num_trainable_var = count_trainable_var()

    def get_params_val(self, sess):
        KB_param_val = get_value_of_valid_tensors(sess, self.cnn_KB_param)
        TS_param_val = [get_value_of_valid_tensors(sess, cnn_TS_param) for cnn_TS_param in self.cnn_TS_param]
        gen_param_val = [get_value_of_valid_tensors(sess, cnn_gen_param) for cnn_gen_param in self.cnn_gen_param]
        fc_param_val = [get_value_of_valid_tensors(sess, fc_param) for fc_param in self.fc_param]

        parameters_val = {}
        parameters_val['conv_KB'] = savemat_wrapper(KB_param_val)
        parameters_val['conv_TS'] = savemat_wrapper_nested_list(TS_param_val)
        parameters_val['conv_generated_weights'] = savemat_wrapper_nested_list(gen_param_val)
        parameters_val['fc_weights'] = savemat_wrapper_nested_list(fc_param_val)
        return parameters_val