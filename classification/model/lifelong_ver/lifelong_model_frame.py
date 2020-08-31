import sys
from abc import abstractmethod
import numpy as np
import tensorflow as tf

_tf_ver = tf.__version__.split('.')
_up_to_date_tf = int(_tf_ver[0]) > 1 or (int(_tf_ver[0])==1 and int(_tf_ver[1]) > 14)
if _up_to_date_tf:
    _tf_tensor = tf.is_tensor
else:
    _tf_tensor = tf.contrib.framework.is_tensor

if sys.version_info.major < 3:
    from utils import count_trainable_var
else:
    from utils.utils import count_trainable_var


class Lifelong_Model_Frame():
    def __init__(self, model_hyperpara, train_hyperpara):
        self.input_size = model_hyperpara['image_dimension']    ## img_width * img_height * img_channel
        self.cnn_channels_size = [self.input_size[-1]]+list(model_hyperpara['channel_sizes'])    ## include dim of input channel
        self.cnn_kernel_size = model_hyperpara['kernel_sizes']     ## len(kernel_size) == 2*(len(channels_size)-1)
        self.cnn_stride_size = model_hyperpara['stride_sizes']
        self.pool_size = model_hyperpara['pooling_size']      ## len(pool_size) == 2*(len(channels_size)-1)
        self.fc_size = model_hyperpara['hidden_layer']
        self.padding_type = model_hyperpara['padding_type']
        self.max_pooling = model_hyperpara['max_pooling']
        self.dropout = model_hyperpara['dropout']
        self.skip_connect = model_hyperpara['skip_connect']

        self.learn_rate = train_hyperpara['lr']
        self.learn_rate_decay = train_hyperpara['lr_decay']
        self.batch_size = model_hyperpara['batch_size']
        self.hidden_act = model_hyperpara['hidden_activation']

        self.num_conv_layers, self.num_fc_layers = len(self.cnn_channels_size)-1, len(self.fc_size)+1

        self.num_tasks = 0
        self.num_trainable_var = 0
        self.output_sizes = []
        self.task_indices = []

    @abstractmethod
    def _build_whole_model(self):
        raise NotImplementedError

    def add_new_task(self, output_dim, curr_task_index, single_input_placeholder=False):
        if self.is_new_task(curr_task_index):
            self.num_tasks += 1
            self.output_sizes.append(output_dim)
            self.task_indices.append(curr_task_index)
            self.task_is_new = True
        else:
            self.task_is_new = False
        self.num_trainable_var = 0
        self.current_task = self.find_task_model(curr_task_index)

        #### placeholder of model
        self.epoch = tf.placeholder(dtype=tf.float32)
        self.dropout_prob = tf.placeholder(dtype=tf.float32)
        if single_input_placeholder:
            self.model_input = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_size[0]*self.input_size[1]*self.input_size[2]])
        else:
            self.model_input = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_size[0]*self.input_size[1]*self.input_size[2]]) for _ in range(self.num_tasks)]
        self.true_output = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size]) for _ in range(self.num_tasks)]

        with tf.name_scope('Minibatch_Data'):
            if single_input_placeholder:
                self.x_batch = [tf.reshape(self.model_input, [-1]+list(self.input_size)) for _ in range(self.num_tasks)]
            else:
                self.x_batch = [tf.reshape(x, [-1]+list(self.input_size)) for x in self.model_input]
            self.y_batch = [y for y in self.true_output]

        self.task_models, self.conv_params, self.fc_params, self.params = [], [], [], []
        self._build_whole_model()
        self.define_eval()
        self.define_loss()
        self.define_accuracy()
        self.define_opt()

    def is_new_task(self, curr_task_index):
        return (not (curr_task_index in self.task_indices))

    def find_task_model(self, task_index_to_search):
        return self.task_indices.index(task_index_to_search)

    def number_of_learned_tasks(self):
        return self.num_tasks

    def define_eval(self):
        with tf.name_scope('Model_Eval'):
            self.eval = [tf.nn.softmax(task_model[-1]) for task_model in self.task_models]
            self.pred = [tf.argmax(task_model[-1], 1) for task_model in self.task_models]

    def define_loss(self):
        with tf.name_scope('Model_Loss'):
            self.loss = [tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(y_batch, tf.int32), logits=task_model[-1])) for y_batch, task_model in zip(self.y_batch, self.task_models)]

    def define_accuracy(self):
        with tf.name_scope('Model_Accuracy'):
            self.accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(task_model[-1], 1), tf.cast(y_batch, tf.int64)), tf.float32)) for y_batch, task_model in zip(self.y_batch, self.task_models)]

    def define_opt(self):
        with tf.name_scope('Optimization'):
            self.grads = tf.gradients(self.loss[self.current_task], self.params[self.current_task])
            trainer = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay))
            self.update = trainer.apply_gradients(list(zip(self.grads, self.params[self.current_task])))

    def convert_tfVar_to_npVar(self, sess):
        if len(self.params) > 0:
            converted_params = []
            for task_params in self.params:
                converted_task_params = []
                for p in task_params:
                    if type(p) == np.ndarray:
                        converted_task_params.append(p)
                    elif _tf_tensor(p):
                        converted_task_params.append(sess.run(p))
                    else:
                        print("\nData type of variable is not based on either TensorFlow or Numpy!!\n")
                        raise ValueError
                converted_params.append(converted_task_params)
            self.np_params = converted_params

    @abstractmethod
    def get_param(self, sess):
        raise NotImplementedError


class Lifelong_Model_EM_Algo_Frame(Lifelong_Model_Frame):
    def __init__(self, model_hyperpara, train_hyperpara):
        super().__init__(model_hyperpara, train_hyperpara)
        self.conv_sharing = []

        def _possible_choices(input_subsets):
            list_subsets = []
            for c in [True, False]:
                for elem in input_subsets:
                    list_subsets.append(elem+[c])
            return list_subsets

        self._possible_configs = [[]]
        for layer_cnt in range(self.num_conv_layers):
            self._possible_configs = _possible_choices(self._possible_configs)
        self.num_possible_configs = len(self._possible_configs)

    @abstractmethod
    def _shared_param_init(self):
        raise NotImplementedError

    def add_new_task(self, output_dim, curr_task_index, single_input_placeholder=False):
        self._shared_param_init()
        super().add_new_task(output_dim, curr_task_index, single_input_placeholder=single_input_placeholder)

    def define_eval(self):
        with tf.name_scope('Model_Eval'):
            self.eval = [tf.nn.softmax(task_model[-1]) for task_model in self.task_models]
            self.pred = [tf.argmax(task_model[-1], 1) for task_model in self.task_models]
            if self.task_is_new:
                self.eval_for_train = [tf.nn.softmax(task_model) for task_model in self.task_models[self.current_task]]
                self.pred_for_train = [tf.argmax(task_model, 1) for task_model in self.task_models[self.current_task]]

                self.likelihood = tf.stack([tf.reduce_prod(tf.boolean_mask(e, tf.one_hot(tf.cast(self.y_batch[self.current_task], tf.int32), self.output_sizes[self.current_task], on_value=True, off_value=False, dtype=tf.bool))) for e in self.eval_for_train])
                self.prior = tf.Variable(np.ones(len(self._possible_configs), dtype=np.float32)/float(len(self._possible_configs)), dtype=tf.float32, trainable=False)
                posterior_tmp = tf.multiply(self.prior, self.likelihood)
                self.posterior = tf.divide(posterior_tmp, tf.reduce_sum(posterior_tmp)+1e-30)

    def define_loss(self):
        with tf.name_scope('Model_Loss'):
            self.loss = [tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(y_batch, tf.int32), logits=task_model[-1])) for y_batch, task_model in zip(self.y_batch, self.task_models)]
            if self.task_is_new:
                self.loss_for_train = [tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(self.y_batch[self.current_task], tf.int32), logits=task_model)) for task_model in self.task_models[self.current_task]]

    def define_accuracy(self):
        with tf.name_scope('Model_Accuracy'):
            self.accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(task_model[-1], 1), tf.cast(y_batch, tf.int64)), tf.float32)) for y_batch, task_model in zip(self.y_batch, self.task_models)]
            if self.task_is_new:
                self.accuracy_for_train = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(task_model, 1), tf.cast(self.y_batch[self.current_task], tf.int64)), tf.float32)) for task_model in self.task_models[self.current_task]]

    def _choose_params_for_sharing_config(self, KB_params, TS_params, conv_params, sharing_configs, num_TS_params_per_layer):
        KB_params_to_return, TS_params_to_return, conv_params_to_return = [], [], []
        for layer_cnt, (c) in enumerate(sharing_configs):
            if c:
                ## sharing
                if KB_params is not None and KB_params[layer_cnt] is not None:
                    KB_params_to_return.append(KB_params[layer_cnt])
                for tmp_cnt in range(num_TS_params_per_layer):
                    if TS_params is not None and TS_params[num_TS_params_per_layer*layer_cnt+tmp_cnt] is not None:
                        TS_params_to_return.append(TS_params[num_TS_params_per_layer*layer_cnt+tmp_cnt])
            elif conv_params is not None:
                ## task-specific
                for tmp_cnt in range(2):
                    if conv_params[2*layer_cnt+tmp_cnt] is not None:
                        conv_params_to_return.append(conv_params[2*layer_cnt+tmp_cnt])
        return KB_params_to_return, TS_params_to_return, conv_params_to_return

    def _choose_params_for_sharing_config_fillNone(self, KB_params, TS_params, conv_params, sharing_configs, num_TS_params_per_layer):
        KB_params_to_return, TS_params_to_return, conv_params_to_return = [], [], []
        for layer_cnt, (c) in enumerate(sharing_configs):
            if c:
                ## sharing
                KB_params_to_return.append(KB_params[layer_cnt])
                TS_params_to_return += [None for _ in range(num_TS_params_per_layer)] if TS_params is None else TS_params[num_TS_params_per_layer*layer_cnt:num_TS_params_per_layer*(layer_cnt+1)]
                conv_params_to_return += [None, None]
            else:
                ## task-specific
                KB_params_to_return.append(None)
                TS_params_to_return += [None for _ in range(num_TS_params_per_layer)]
                conv_params_to_return += conv_params[2*layer_cnt:2*(layer_cnt+1)]
        return KB_params_to_return, TS_params_to_return, conv_params_to_return

    def best_config(self, sess):
        prior_val = sess.run(self.prior)
        return np.argmax(prior_val)

    def _weighted_sum_two_grads(self, g1, g2, w1, w2):
        if g1 is not None and g2 is not None:
            return w1*g1+w2*g2
        elif g1 is not None:
            return w1*g1
        elif g2 is not None:
            return w2*g2
        else:
            return None

    def _weighted_sum_grads(self, grad_list, weights):
        for i in range(1, len(grad_list)):
            if i < 2:
                result = self._weighted_sum_two_grads(grad_list[i-1], grad_list[i], weights[i-1], weights[i])
            elif i < len(grad_list):
                result = self._weighted_sum_two_grads(result, grad_list[i], 1.0, weights[i])
        return result