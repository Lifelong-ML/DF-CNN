import sys
import numpy as np
import tensorflow as tf

if sys.version_info.major < 3:
    from utils import *
    from utils_nn import *
else:
    from utils.utils import *
    from utils.utils_nn import *


############################################################
#####     functions to generate ELLA DNN parameter     #####
############################################################
#### function to generate knowledge-base parameters for ELLA_tensorfactor layer
def new_KB_param(shape, layer_number, task_number):
    kb_name = 'KB_'+str(layer_number)+'_'+str(task_number)
    return tf.get_variable(name=kb_name, shape=shape, dtype=tf.float32)

#### function to generate task-specific parameters for ELLA_tensorfactor layer
def new_dfcnn_direct_TS_param(shape, layer_number, task_number):
    ts_w_name, ts_b_name, ts_p_name = 'TS_DeconvW0_'+str(layer_number)+'_'+str(task_number), 'TS_Deconvb0_'+str(layer_number)+'_'+str(task_number), 'TS_Convb0_'+str(layer_number)+'_'+str(task_number)
    return [tf.get_variable(name=ts_w_name, shape=shape[0], dtype=tf.float32), tf.get_variable(name=ts_b_name, shape=shape[1], dtype=tf.float32), tf.get_variable(name=ts_p_name, shape=shape[2], dtype=tf.float32)]

#### function to generate task-specific parameters for ELLA_tensorfactor layer
def new_dfcnn_TS_param(shape, layer_number, task_number):
    ts_w_name, ts_b_name, ts_k_name, ts_p_name = 'TS_DeconvW0_'+str(layer_number)+'_'+str(task_number), 'TS_Deconvb0_'+str(layer_number)+'_'+str(task_number), 'TS_ConvW1_'+str(layer_number)+'_'+str(task_number), 'TS_Convb0_'+str(layer_number)+'_'+str(task_number)
    return [tf.get_variable(name=ts_w_name, shape=shape[0], dtype=tf.float32), tf.get_variable(name=ts_b_name, shape=shape[1], dtype=tf.float32), tf.get_variable(name=ts_k_name, shape=shape[2], dtype=tf.float32), tf.get_variable(name=ts_p_name, shape=shape[3], dtype=tf.float32)]

#### function to generate task-specific parameters for ELLA_tensorfactor layer
def new_dfcnn_tc2_TS_param(shape, layer_number, task_number):
    ts_w_name, ts_b_name, ts_k_name, ts_k_name2, ts_p_name = 'TS_DeconvW0_'+str(layer_number)+'_'+str(task_number), 'TS_Deconvb0_'+str(layer_number)+'_'+str(task_number), 'TS_tdot_W1_'+str(layer_number)+'_'+str(task_number), 'TS_tdot_W2_'+str(layer_number)+'_'+str(task_number), 'TS_tdot_b0_'+str(layer_number)+'_'+str(task_number)
    return [tf.get_variable(name=ts_w_name, shape=shape[0], dtype=tf.float32), tf.get_variable(name=ts_b_name, shape=shape[1], dtype=tf.float32), tf.get_variable(name=ts_k_name, shape=shape[2], dtype=tf.float32), tf.get_variable(name=ts_k_name2, shape=shape[3], dtype=tf.float32), tf.get_variable(name=ts_p_name, shape=shape[4], dtype=tf.float32)]



###############################################################
##### functions for adding ELLA network (CNN/Deconv ver)  #####
###############################################################

#### function to generate convolutional layer with shared knowledge base
#### KB_size : [filter_height(and width), num_of_channel]
#### TS_size : deconv_filter_height(and width)
#### TS_stride_size : [stride_in_height, stride_in_width]
def new_dfcnn_direct_layer(layer_input, k_size, ch_size, stride_size, KB_size, TS_size, TS_stride_size, layer_num, task_num, activation_fn=tf.nn.relu, para_activation_fn=tf.nn.relu, KB_param=None, TS_param=None, padding_type='SAME', max_pool=False, pool_size=None):
    assert (k_size[0] == k_size[1] and k_size[0] == (KB_size[0]-1)*TS_stride_size[0]+1), "CNN kernel size does not match the output size of Deconv from KB"

    with tf.name_scope('ELLA_cdnn_KB'):
        if KB_param is None:
            ## KB \in R^{1 \times h \times w \times c}
            KB_param = new_KB_param([1, KB_size[0], KB_size[0], KB_size[1]], layer_num, task_num)
        if TS_param is None:
            ## TS1 : Deconv W \in R^{h \times w \times ch_in*ch_out \times c}
            ## TS2 : Deconv bias \in R^{ch_out}
            TS_param = new_dfcnn_direct_TS_param([[TS_size, TS_size, ch_size[0]*ch_size[1], KB_size[1]], [1, 1, 1, ch_size[0]*ch_size[1]], [1, 1, 1, ch_size[1]]], layer_num, task_num)

    with tf.name_scope('ELLA_cdnn_TS'):
        para_tmp = tf.add(tf.nn.conv2d_transpose(KB_param, TS_param[0], [1, k_size[0], k_size[1], ch_size[0]*ch_size[1]], strides=[1, TS_stride_size[0], TS_stride_size[1], 1]), TS_param[1])
        if para_activation_fn is not None:
            para_tmp = para_activation_fn(para_tmp)

        W, b = tf.reshape(para_tmp, k_size+ch_size), TS_param[2]

    layer_eqn, _ = new_cnn_layer(layer_input, k_size+ch_size, stride_size=stride_size, activation_fn=activation_fn, weight=W, bias=b, padding_type=padding_type, max_pooling=max_pool, pool_size=pool_size)
    return layer_eqn, [KB_param], TS_param, [W, b]


#### function to generate network of convolutional layers with shared knowledge base
def new_dfcnn_direct_net(net_input, k_sizes, ch_sizes, stride_sizes, KB_sizes, TS_sizes, TS_stride_sizes, activation_fn=tf.nn.relu, para_activation_fn=tf.nn.relu, KB_params=None, TS_params=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, flat_output=False, task_index=0):
    _num_TS_param_per_layer = 3

    ## first element : make new KB&TS / second element : make new TS / third element : not make new para
    control_flag = [(KB_params is None and TS_params is None), (not (KB_params is None) and (TS_params is None)), not (KB_params is None or TS_params is None)]
    if control_flag[1]:
        TS_params = []
    elif control_flag[0]:
        KB_params, TS_params = [], []
    cnn_gen_params=[]

    with tf.name_scope('ELLA_cdnn_net'):
        layers = []
        for layer_cnt in range(len(k_sizes)//2):
            if layer_cnt == 0 and control_flag[0]:
                layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp = new_dfcnn_direct_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[layer_cnt], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=None, TS_param=None, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1])
            elif layer_cnt == 0 and control_flag[1]:
                layer_tmp, _, TS_para_tmp, cnn_gen_para_tmp = new_dfcnn_direct_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[layer_cnt], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=None, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1])
            elif layer_cnt == 0 and control_flag[2]:
                layer_tmp, _, _, cnn_gen_para_tmp = new_dfcnn_direct_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[layer_cnt], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1])
            elif control_flag[0]:
                layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp = new_dfcnn_direct_layer(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[layer_cnt], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=None, TS_param=None, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1])
            elif control_flag[1]:
                layer_tmp, _, TS_para_tmp, cnn_gen_para_tmp = new_dfcnn_direct_layer(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[layer_cnt], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=None, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1])
            elif control_flag[2]:
                layer_tmp, _, _, cnn_gen_para_tmp = new_dfcnn_direct_layer(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[layer_cnt], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1])

            layers.append(layer_tmp)
            cnn_gen_params = cnn_gen_params + cnn_gen_para_tmp
            if control_flag[1]:
                TS_params = TS_params + TS_para_tmp
            elif control_flag[0]:
                KB_params = KB_params + KB_para_tmp
                TS_params = TS_params + TS_para_tmp

        #### flattening output
        if flat_output:
            output_dim = [int(layers[-1].shape[1]*layers[-1].shape[2]*layers[-1].shape[3])]
            layers.append(tf.reshape(layers[-1], [-1, output_dim[0]]))
        else:
            output_dim = layers[-1].shape[1:]

        #### add dropout layer
        if dropout:
            layers.append(tf.nn.dropout(layers[-1], dropout_prob))
    return (layers, KB_params, TS_params, cnn_gen_params, output_dim)


#### function to generate network of cnn->ffnn
def new_dfcnn_direct_fc_net(net_input, k_sizes, ch_sizes, stride_sizes, fc_sizes, cnn_KB_sizes, cnn_TS_sizes, cnn_TS_stride_sizes, cnn_activation_fn=tf.nn.relu, cnn_para_activation_fn=tf.nn.relu, cnn_KB_params=None, cnn_TS_params=None, fc_activation_fn=tf.nn.relu, fc_params=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, output_type=None, task_index=0):
    ## add CNN layers
    cnn_model, cnn_KB_params, cnn_TS_params, cnn_gen_params, cnn_output_dim = new_dfcnn_direct_net(net_input, k_sizes, ch_sizes, stride_sizes, cnn_KB_sizes, cnn_TS_sizes, cnn_TS_stride_sizes, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_params=cnn_KB_params, TS_params=cnn_TS_params, padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, dropout=dropout, dropout_prob=dropout_prob, flat_output=True, task_index=task_index)

    ## add fc layers
    fc_model, fc_params = new_fc_net(cnn_model[-1], fc_sizes, activation_fn=fc_activation_fn, params=fc_params, output_type=output_type, tensorboard_name_scope='fc_net')

    return (cnn_model+fc_model, cnn_KB_params, cnn_TS_params, cnn_gen_params, fc_params)



###########################################################################
##### functions for adding ELLA network (CNN/Deconv & Tensordot ver)  #####
###########################################################################

#### KB_size : [filter_height(and width), num_of_channel]
#### TS_size : [deconv_filter_height(and width), deconv_filter_channel]
#### TS_stride_size : [stride_in_height, stride_in_width]
def new_dfcnn_layer(layer_input, k_size, ch_size, stride_size, KB_size, TS_size, TS_stride_size, layer_num, task_num, activation_fn=tf.nn.relu, para_activation_fn=tf.nn.relu, KB_param=None, TS_param=None, padding_type='SAME', max_pool=False, pool_size=None):
    assert (k_size[0] == k_size[1] and k_size[0] == (KB_size[0]-1)*TS_stride_size[0]+1), "CNN kernel size does not match the output size of Deconv from KB"

    with tf.name_scope('ELLA_cdnn_KB'):
        if KB_param is None:
            ## KB \in R^{1 \times h \times w \times c}
            KB_param = new_KB_param([1, KB_size[0], KB_size[0], KB_size[1]], layer_num, task_num)
        if TS_param is None:
            ## TS1 : Deconv W \in R^{h \times w \times kb_c_out \times c}
            ## TS2 : Deconv bias \in R^{kb_c_out}
            ## TS3 : tensor W \in R^{kb_c_out \times ch_in \times ch_out}
            ## TS4 : Conv bias \in R^{ch_out}
            TS_param = new_dfcnn_TS_param([[TS_size[0], TS_size[0], TS_size[1], KB_size[1]], [1, 1, 1, TS_size[1]], [TS_size[1], ch_size[0], ch_size[1]], [1, 1, 1, ch_size[1]]], layer_num, task_num)

    with tf.name_scope('ELLA_cdnn_TS'):
        para_tmp = tf.add(tf.nn.conv2d_transpose(KB_param, TS_param[0], [1, k_size[0], k_size[1], TS_size[1]], strides=[1, TS_stride_size[0], TS_stride_size[1], 1]), TS_param[1])
        para_tmp = tf.reshape(para_tmp, [k_size[0], k_size[1], TS_size[1]])
        if para_activation_fn is not None:
            para_tmp = para_activation_fn(para_tmp)
        W = tf.tensordot(para_tmp, TS_param[2], [[2], [0]])
        b = TS_param[3]

    layer_eqn, _ = new_cnn_layer(layer_input, k_size+ch_size, stride_size=stride_size, activation_fn=activation_fn, weight=W, bias=b, padding_type=padding_type, max_pooling=max_pool, pool_size=pool_size)
    return layer_eqn, [KB_param], TS_param, [W, b]


#### function to generate network of convolutional layers with shared knowledge base
def new_dfcnn_net(net_input, k_sizes, ch_sizes, stride_sizes, KB_sizes, TS_sizes, TS_stride_sizes, activation_fn=tf.nn.relu, para_activation_fn=tf.nn.relu, KB_params=None, TS_params=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, flat_output=False, task_index=0):
    _num_TS_param_per_layer = 4

    ## first element : make new KB&TS / second element : make new TS / third element : not make new para / fourth element : make new KB
    control_flag = [(KB_params is None and TS_params is None), (not (KB_params is None) and (TS_params is None)), not (KB_params is None or TS_params is None), ((KB_params is None) and not (TS_params is None))]
    if control_flag[1]:
        TS_params = []
    elif control_flag[3]:
        KB_params = []
    elif control_flag[0]:
        KB_params, TS_params = [], []
    cnn_gen_params = []

    with tf.name_scope('ELLA_cdnn_net'):
        layers = []
        for layer_cnt in range(len(k_sizes)//2):
            if layer_cnt == 0 and control_flag[0]:
                layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp = new_dfcnn_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=None, TS_param=None, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1])
            elif layer_cnt == 0 and control_flag[1]:
                layer_tmp, _, TS_para_tmp, cnn_gen_para_tmp = new_dfcnn_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=None, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1])
            elif layer_cnt == 0 and control_flag[2]:
                layer_tmp, _, _, cnn_gen_para_tmp = new_dfcnn_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1])
            elif layer_cnt == 0 and control_flag[3]:
                layer_tmp, KB_para_tmp, _, cnn_gen_para_tmp = new_dfcnn_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=None, TS_param=TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1])
            elif control_flag[0]:
                layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp = new_dfcnn_layer(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=None, TS_param=None, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1])
            elif control_flag[1]:
                layer_tmp, _, TS_para_tmp, cnn_gen_para_tmp = new_dfcnn_layer(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=None, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1])
            elif control_flag[2]:
                layer_tmp, _, _, cnn_gen_para_tmp = new_dfcnn_layer(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1])
            elif control_flag[3]:
                layer_tmp, KB_para_tmp, _, cnn_gen_para_tmp = new_dfcnn_layer(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=None, TS_param=TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1])

            layers.append(layer_tmp)
            cnn_gen_params = cnn_gen_params + cnn_gen_para_tmp
            if control_flag[1]:
                TS_params = TS_params + TS_para_tmp
            elif control_flag[3]:
                KB_params = KB_params + KB_para_tmp
            elif control_flag[0]:
                KB_params = KB_params + KB_para_tmp
                TS_params = TS_params + TS_para_tmp

        #### flattening output
        if flat_output:
            output_dim = [int(layers[-1].shape[1]*layers[-1].shape[2]*layers[-1].shape[3])]
            layers.append(tf.reshape(layers[-1], [-1, output_dim[0]]))
        else:
            output_dim = layers[-1].shape[1:]

        #### add dropout layer
        if dropout:
            layers.append(tf.nn.dropout(layers[-1], dropout_prob))
    return (layers, KB_params, TS_params, cnn_gen_params, output_dim)


#### function to generate network of cnn (with shared KB through deconv)-> simple ffnn
def new_dfcnn_fc_net(net_input, k_sizes, ch_sizes, stride_sizes, fc_sizes, cnn_KB_sizes, cnn_TS_sizes, cnn_TS_stride_sizes, cnn_activation_fn=tf.nn.relu, cnn_para_activation_fn=tf.nn.relu, cnn_KB_params=None, cnn_TS_params=None, fc_activation_fn=tf.nn.relu, fc_params=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, output_type=None, task_index=0):
    ## add CNN layers
    cnn_model, cnn_KB_params, cnn_TS_params, cnn_gen_params, cnn_output_dim = new_dfcnn_net(net_input, k_sizes, ch_sizes, stride_sizes, cnn_KB_sizes, cnn_TS_sizes, cnn_TS_stride_sizes, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_params=cnn_KB_params, TS_params=cnn_TS_params, padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, dropout=dropout, dropout_prob=dropout_prob, flat_output=True, task_index=task_index)

    ## add fc layers
    fc_model, fc_params = new_fc_net(cnn_model[-1], fc_sizes, activation_fn=fc_activation_fn, params=fc_params, output_type=output_type, tensorboard_name_scope='fc_net')

    return (cnn_model+fc_model, cnn_KB_params, cnn_TS_params, cnn_gen_params, fc_params)



###########################################################################
##### functions for adding ELLA network (CNN/Deconv & Tensordot ver2)  #####
###########################################################################

#### KB_size : [filter_height(and width), num_of_channel0, num_of_channel1]
#### TS_size : [deconv_filter_height(and width), deconv_filter_channel]
#### TS_stride_size : [stride_in_height, stride_in_width]
def new_dfcnn_tc2_layer(layer_input, k_size, ch_size, stride_size, KB_size, TS_size, TS_stride_size, layer_num, task_num, activation_fn=tf.nn.relu, para_activation_fn=tf.nn.relu, KB_param=None, TS_param=None, padding_type='SAME', max_pool=False, pool_size=None):
    assert (k_size[0] == k_size[1] and k_size[0] == (KB_size[0]-1)*TS_stride_size[0]+1), "CNN kernel size does not match the output size of Deconv from KB"

    with tf.name_scope('ELLA_cdnn_KB'):
        if KB_param is None:
            ## KB \in R^{d \times h \times w \times c}
            KB_param = new_KB_param([KB_size[1], KB_size[0], KB_size[0], KB_size[2]], layer_num, task_num)
        if TS_param is None:
            ## TS1 : Deconv W \in R^{h \times w \times kb_c_out \times c}
            ## TS2 : Deconv bias \in R^{kb_c_out}
            ## TS3 : tensor W \in R^{d \times ch_in}
            ## TS4 : tensor W \in R^{kb_c_out \times ch_out}
            ## TS5 : Conv bias \in R^{ch_out}
            TS_param = new_dfcnn_tc2_TS_param([[TS_size[0], TS_size[0], TS_size[1], KB_size[2]], [1, 1, 1, TS_size[1]], [KB_size[1], ch_size[0]], [TS_size[1], ch_size[1]], [1, 1, 1, ch_size[1]]], layer_num, task_num)

    with tf.name_scope('ELLA_cdnn_TS'):
        para_tmp = tf.add(tf.nn.conv2d_transpose(KB_param, TS_param[0], [KB_size[1], k_size[0], k_size[1], TS_size[1]], strides=[1, TS_stride_size[0], TS_stride_size[1], 1]), TS_param[1])
        if para_activation_fn is not None:
            para_tmp = para_activation_fn(para_tmp)
        para_tmp = tf.tensordot(para_tmp, TS_param[2], [[0], [0]])
        W = tf.tensordot(para_tmp, TS_param[3], [[2], [0]])
        b = TS_param[4]

    layer_eqn, _ = new_cnn_layer(layer_input, k_size+ch_size, stride_size=stride_size, activation_fn=activation_fn, weight=W, bias=b, padding_type=padding_type, max_pooling=max_pool, pool_size=pool_size)
    return layer_eqn, [KB_param], TS_param, [W, b]


#### function to generate network of convolutional layers with shared knowledge base
def new_dfcnn_tc2_net(net_input, k_sizes, ch_sizes, stride_sizes, KB_sizes, TS_sizes, TS_stride_sizes, activation_fn=tf.nn.relu, para_activation_fn=tf.nn.relu, KB_params=None, TS_params=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, flat_output=False, task_index=0):
    _num_TS_param_per_layer = 5

    ## first element : make new KB&TS / second element : make new TS / third element : not make new para / fourth element : make new KB
    control_flag = [(KB_params is None and TS_params is None), (not (KB_params is None) and (TS_params is None)), not (KB_params is None or TS_params is None), ((KB_params is None) and not (TS_params is None))]
    if control_flag[1]:
        TS_params = []
    elif control_flag[3]:
        KB_params = []
    elif control_flag[0]:
        KB_params, TS_params = [], []
    cnn_gen_params = []

    with tf.name_scope('ELLA_cdnn_net'):
        layers = []
        for layer_cnt in range(len(k_sizes)//2):
            if layer_cnt == 0 and control_flag[0]:
                layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp = new_dfcnn_tc2_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[3*layer_cnt:3*(layer_cnt+1)], TS_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=None, TS_param=None, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1])
            elif layer_cnt == 0 and control_flag[1]:
                layer_tmp, _, TS_para_tmp, cnn_gen_para_tmp = new_dfcnn_tc2_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[3*layer_cnt:3*(layer_cnt+1)], TS_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=None, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1])
            elif layer_cnt == 0 and control_flag[2]:
                layer_tmp, _, _, cnn_gen_para_tmp = new_dfcnn_tc2_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[3*layer_cnt:3*(layer_cnt+1)], TS_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1])
            elif layer_cnt == 0 and control_flag[3]:
                layer_tmp, KB_para_tmp, _, cnn_gen_para_tmp = new_dfcnn_tc2_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[3*layer_cnt:3*(layer_cnt+1)], TS_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=None, TS_param=TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1])
            elif control_flag[0]:
                layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp = new_dfcnn_tc2_layer(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[3*layer_cnt:3*(layer_cnt+1)], TS_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=None, TS_param=None, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1])
            elif control_flag[1]:
                layer_tmp, _, TS_para_tmp, cnn_gen_para_tmp = new_dfcnn_tc2_layer(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[3*layer_cnt:3*(layer_cnt+1)], TS_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=None, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1])
            elif control_flag[2]:
                layer_tmp, _, _, cnn_gen_para_tmp = new_dfcnn_tc2_layer(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[3*layer_cnt:3*(layer_cnt+1)], TS_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1])
            elif control_flag[3]:
                layer_tmp, KB_para_tmp, _, cnn_gen_para_tmp = new_dfcnn_tc2_layer(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[3*layer_cnt:3*(layer_cnt+1)], TS_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=None, TS_param=TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1])

            layers.append(layer_tmp)
            cnn_gen_params = cnn_gen_params + cnn_gen_para_tmp
            if control_flag[1]:
                TS_params = TS_params + TS_para_tmp
            elif control_flag[3]:
                KB_params = KB_params + KB_para_tmp
            elif control_flag[0]:
                KB_params = KB_params + KB_para_tmp
                TS_params = TS_params + TS_para_tmp

        #### flattening output
        if flat_output:
            output_dim = [int(layers[-1].shape[1]*layers[-1].shape[2]*layers[-1].shape[3])]
            layers.append(tf.reshape(layers[-1], [-1, output_dim[0]]))
        else:
            output_dim = layers[-1].shape[1:]

        #### add dropout layer
        if dropout:
            layers.append(tf.nn.dropout(layers[-1], dropout_prob))
    return (layers, KB_params, TS_params, cnn_gen_params, output_dim)


#### function to generate network of cnn (with shared KB through deconv)-> simple ffnn
def new_dfcnn_tc2_fc_net(net_input, k_sizes, ch_sizes, stride_sizes, fc_sizes, cnn_KB_sizes, cnn_TS_sizes, cnn_TS_stride_sizes, cnn_activation_fn=tf.nn.relu, cnn_para_activation_fn=tf.nn.relu, cnn_KB_params=None, cnn_TS_params=None, fc_activation_fn=tf.nn.relu, fc_params=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, output_type=None, task_index=0):
    ## add CNN layers
    cnn_model, cnn_KB_params, cnn_TS_params, cnn_gen_params, cnn_output_dim = new_dfcnn_tc2_net(net_input, k_sizes, ch_sizes, stride_sizes, cnn_KB_sizes, cnn_TS_sizes, cnn_TS_stride_sizes, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_params=cnn_KB_params, TS_params=cnn_TS_params, padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, dropout=dropout, dropout_prob=dropout_prob, flat_output=True, task_index=task_index)

    ## add fc layers
    fc_model, fc_params = new_fc_net(cnn_model[-1], fc_sizes, activation_fn=fc_activation_fn, params=fc_params, output_type=output_type, tensorboard_name_scope='fc_net')

    return (cnn_model+fc_model, cnn_KB_params, cnn_TS_params, cnn_gen_params, fc_params)