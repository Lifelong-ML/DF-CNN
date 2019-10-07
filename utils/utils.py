import numpy as np
import tensorflow as tf
from tensorflow import trainable_variables
import sys


##############################################
#####  functions to save NN's parameter  #####
##############################################
def get_list_of_valid_tensors(list_of_variables):
    list_of_valid_tensors = []
    for elem in list_of_variables:
        if elem is not None:
            list_of_valid_tensors.append(elem)
    return list_of_valid_tensors

def get_value_of_valid_tensors(tf_sess, list_of_variables):
    list_of_val = []
    for elem in list_of_variables:
        list_of_val.append(elem if (elem is None) else tf_sess.run(elem))
    return list_of_val

def savemat_wrapper(list_of_data):
    data_to_save = np.zeros((len(list_of_data),), dtype=np.object)
    for cnt in range(len(list_of_data)):
        if list_of_data[cnt] is not None:
            data_to_save[cnt] = list_of_data[cnt]
    return data_to_save

def savemat_wrapper_nested_list(list_of_data):
    data_to_save = np.zeros((len(list_of_data),), dtype=np.object)
    for cnt in range(len(list_of_data)):
        data_to_save[cnt] = savemat_wrapper(list_of_data[cnt])
    return data_to_save


############################################
#### functions for (MTL) model's output ####
############################################
def mtl_model_output_functions(models, y_batches, num_tasks):
    with tf.name_scope('Model_Eval'):
        train_eval = [tf.nn.softmax(models[0][x][-1]) for x in range(num_tasks)]
        valid_eval = [tf.nn.softmax(models[1][x][-1]) for x in range(num_tasks)]
        test_eval = [tf.nn.softmax(models[2][x][-1]) for x in range(num_tasks)]

        train_output_label = [tf.argmax(models[0][x][-1], 1) for x in range(num_tasks)]
        valid_output_label = [tf.argmax(models[1][x][-1], 1) for x in range(num_tasks)]
        test_output_label = [tf.argmax(models[2][x][-1], 1) for x in range(num_tasks)]

    with tf.name_scope('Model_Loss'):
        train_loss = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(y_batches[0][x], tf.int32), logits=models[0][x][-1]) for x in range(num_tasks)]
        valid_loss = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(y_batches[1][x], tf.int32), logits=models[1][x][-1]) for x in range(num_tasks)]
        test_loss = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(y_batches[2][x], tf.int32), logits=models[2][x][-1]) for x in range(num_tasks)]

    with tf.name_scope('Model_Accuracy'):
        train_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(models[0][x][-1], 1), tf.cast(y_batches[0][x], tf.int64)), tf.float32)) for x in range(num_tasks)]
        valid_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(models[1][x][-1], 1), tf.cast(y_batches[1][x], tf.int64)), tf.float32)) for x in range(num_tasks)]
        test_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(models[2][x][-1], 1), tf.cast(y_batches[2][x], tf.int64)), tf.float32)) for x in range(num_tasks)]
    return (train_eval, valid_eval, test_eval, train_loss, valid_loss, test_loss, train_accuracy, valid_accuracy, test_accuracy, train_output_label, valid_output_label, test_output_label)


############################################
####   Basic functions for Neural Net   ####
############################################
_weight_init_stddev = 0.05

#### Leaky ReLu function
def leaky_relu(function_in, leaky_alpha=0.01):
    return tf.nn.relu(function_in) - leaky_alpha*tf.nn.relu(-function_in)

#### function to generate weight parameter
def new_weight(shape, trainable=True, init_tensor=None):
    return tf.Variable(tf.truncated_normal(shape, stddev=_weight_init_stddev) if init_tensor is None else init_tensor, trainable=trainable)

#### function to generate bias parameter
def new_bias(shape, trainable=True, init_val=0.2, init_tensor=None):
    return tf.Variable(tf.constant(init_val, dtype=tf.float32, shape=shape) if init_tensor is None else init_tensor, trainable=trainable)

#### function to count trainable parameters in computational graph
def count_trainable_var():
    total_para_cnt = 0
    for variable in trainable_variables():
        para_cnt_tmp = 1
        for dim in variable.get_shape():
            para_cnt_tmp = para_cnt_tmp * dim.value
        total_para_cnt = total_para_cnt + para_cnt_tmp
    return total_para_cnt