## Lifelong learning with dynamically expandable networks : https://arxiv.org/abs/1708.01547

import tensorflow as tf
import numpy as np
import re, random, collections
from collections import defaultdict
from numpy import linalg as LA
#from sklearn.metrics import roc_curve, auc

from os import getcwd, listdir, mkdir
import scipy.io as spio

#from tensorflow.contrib.memory_stats.python.ops import memory_stats_ops

from utils.utils import convert_array_to_oneHot

_num_tasks = 10


def save_param_to_mat(param_dict):
    param_to_save_format = {}
    for key, val in param_dict.items():
        scope_name = key.split(':')[0]
        scope = scope_name.split('/')[0]
        name = scope_name.split('/')[1]
        new_scope_name = scope + '_' + name
        param_to_save_format[new_scope_name] = val
    return param_to_save_format


def accuracy(preds, labels):
    return (100.0 * np.sum(np.argmax(preds, 1) == np.argmax(labels, 1)) / preds.shape[0])

def RMSE(p, y):
    N = p.shape[0]
    diff = p - y
    return np.sqrt((diff**2).mean())

'''
def ROC_AUC(p, y):
    fpr, tpr, th = roc_curve(y, p)
    _auc = auc(fpr, tpr)
    _roc = (fpr, tpr)
    return _roc, _auc
'''

class CNN_FC_DEN(object):
    def __init__(self, model_hyperpara, train_hyperpara, data_info):
        print("This code highly depends on the authors' code which isn't published to public yet (Obtained code directly from authors). The remaining part is removed, and recommend to contact DEN authors for the code\n\n")
        raise NotImplementedError

        self.T = 0
        self.task_indices = []

        self.params = dict()
        self.param_trained = set()
        self.time_stamp = dict()

        self.batch_size = model_hyperpara['batch_size']
        self.input_shape = model_hyperpara['image_dimension']
        self.n_classes = data_info[2][0]
        self.ex_k = model_hyperpara['den_expansion']

        self.n_conv_layers = len(model_hyperpara['channel_sizes'])
        self.n_fc_layers = len(model_hyperpara['hidden_layer'])+1
        self.n_layers = self.n_conv_layers + self.n_fc_layers
        self.layer_types = ['input']+['conv' for _ in range(self.n_conv_layers)]+['fc' for _ in range(self.n_fc_layers)]

        self.cnn_kernel = model_hyperpara['kernel_sizes']
        self.cnn_stride = model_hyperpara['stride_sizes']
        self.cnn_channel = [self.input_shape[-1]]+model_hyperpara['channel_sizes']
        self.cnn_max_pool = model_hyperpara['max_pooling']
        self.cnn_pooling_sizes = model_hyperpara['pooling_size']
        self.cnn_dropout = model_hyperpara['dropout']
        self.cnn_padding_type = model_hyperpara['padding_type']
        self.cnn_output_dim = self.get_cnn_output_dims()
        self.cnn_output_spatial_dim = self.cnn_output_dim//self.cnn_channel[-1]
        self.fc_hiddens = [self.cnn_output_dim] + model_hyperpara['hidden_layer'] + [self.n_classes]

        self.l1_lambda = model_hyperpara['l1_lambda']
        self.l2_lambda = model_hyperpara['l2_lambda']
        self.gl_lambda = model_hyperpara['gl_lambda']
        self.regular_lambda = model_hyperpara['reg_lambda']
        self.loss_thr = model_hyperpara['loss_threshold']
        self.spl_thr = model_hyperpara['sparsity_threshold']
        self.scale_up = 15

        self.init_lr = train_hyperpara['lr']
        self.lr_decay = train_hyperpara['lr_decay']
        #self.max_iter = train_hyperpara['learning_step_max']
        #self.early_training = self.max_iter / 10.
        self.max_epoch_per_task = train_hyperpara['patience']
        self.num_training_epoch = 0
        self.train_iter_counter = 0
        self.task_change_epoch = [1]

        for i in range(self.n_layers-1):
            if self.layer_types[i+1] == 'conv':
                w = self.create_variable('layer%d'%(i+1), 'weight', self.cnn_kernel[2*i:2*(i+1)]+self.cnn_channel[i:i+2])
                b = self.create_variable('layer%d'%(i+1), 'biases', [self.cnn_channel[i+1]])
            elif self.layer_types[i+1] == 'fc':
                w = self.create_variable('layer%d'%(i+1), 'weight', [self.fc_hiddens[i-self.n_conv_layers], self.fc_hiddens[i+1-self.n_conv_layers]])
                b = self.create_variable('layer%d'%(i+1), 'biases', [self.fc_hiddens[i+1-self.n_conv_layers]])
            else:
                continue

        self.cur_W, self.prev_W = dict(), dict()

    def set_sess_config(self, config):
        self.sess_config = config

    def get_cnn_output_dims(self):
        raise NotImplementedError

    def get_params(self):
        """ Access the parameters """
        raise NotImplementedError

    def load_params(self, params, top = False, time = 999):
        """ parmas: it contains weight parameters used in network, like ckpt """
        raise NotImplementedError

    def num_trainable_var(self, params_list=None):
        raise NotImplementedError

    def create_variable(self, scope, name, shape, trainable = True):
        raise NotImplementedError

    def get_variable(self, scope, name, trainable = True):
        raise NotImplementedError

    def extend_bottom(self, scope, ex_k = 10):
        """ bottom layer expansion. scope is range of layer """
        raise NotImplementedError

    def extend_top(self, scope, ex_k = 10):
        """ top layer expansion. scope is range of layer """
        raise NotImplementedError

    def extend_param(self, scope, ex_k, l):
        raise NotImplementedError

    def build_model(self, task_id, prediction = False, splitting = False, expansion = None):
        raise NotImplementedError

    def selective_learning(self, task_id, selected_params):
        raise NotImplementedError

    def optimization(self, prev_W, selective = False, splitting = False, expansion = None):
        raise NotImplementedError

    def set_initial_states(self, decay_step, g_step_init=0.0):
        raise NotImplementedError

    def add_task(self, task_id, data, saveParam=False, saveGraph=False):
        raise NotImplementedError

    def run_epoch(self, opt, loss, data, desc = 'Train', selective = False, s_iter = 0, s_max=-1):
        raise NotImplementedError

    def compute_accuracy_during_training(self, data):
        raise NotImplementedError

    def reformat_accuracy_history(self, acc_dest, acc_src):
        raise NotImplementedError

    def data_iteration(self, X, Y, desc = 'Train'):
        raise NotImplementedError

    def get_performance(self, p, y):
        raise NotImplementedError

    def predict_perform(self, task_id, X, Y, task_name = None):
        raise NotImplementedError

    def prediction(self, X):
        raise NotImplementedError

    def destroy_graph(self):
        tf.reset_default_graph()

    def avg_sparsity(self, task_id):
        raise NotImplementedError
