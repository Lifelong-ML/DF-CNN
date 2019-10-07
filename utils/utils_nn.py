import sys
import numpy as np
import tensorflow as tf

if sys.version_info.major < 3:
    from utils import *
    from utils_tensor_factorization import TensorProducer
else:
    from utils.utils import *
    from utils.utils_tensor_factorization import TensorProducer

############################################################
#####   functions for adding fully-connected network   #####
############################################################
#### function to add fully-connected layer
def new_fc_layer(layer_input, output_dim, activation_fn=tf.nn.relu, weight=None, bias=None, trainable=True, use_numpy_var_in_graph=False):
    input_dim = int(layer_input.shape[1])
    with tf.name_scope('fc_layer'):
        if weight is None:
            weight = new_weight(shape=[input_dim, output_dim], trainable=trainable)
        elif (type(weight) == np.ndarray) and not use_numpy_var_in_graph:
            weight = new_weight(shape=[input_dim, output_dim], init_tensor=weight, trainable=trainable)
        if bias is None:
            bias = new_bias(shape=[output_dim], trainable=trainable)
        elif (type(bias) == np.ndarray) and not use_numpy_var_in_graph:
            bias = new_bias(shape=[output_dim], init_tensor=bias, trainable=trainable)

        if activation_fn is None:
            layer = tf.matmul(layer_input, weight) + bias
        else:
            layer = activation_fn( tf.matmul(layer_input, weight) + bias )
    return layer, [weight, bias]

#### function to generate network of fully-connected layers
####      'dim_layers' contains input/output layer
def new_fc_net(net_input, dim_layers, activation_fn=tf.nn.relu, params=None, output_type=None, tensorboard_name_scope='fc_net', trainable=True, use_numpy_var_in_graph=False):
    if params is None:
        params = [None for _ in range(2*len(dim_layers))]

    layers, params_to_return = [], []
    if len(dim_layers) < 1:
        #### for the case that hard-parameter shared network does not have shared layers
        layers.append(net_input)
    else:
        with tf.name_scope(tensorboard_name_scope):
            for cnt in range(len(dim_layers)):
                if cnt == 0:
                    layer_tmp, param_tmp = new_fc_layer(net_input, dim_layers[cnt], activation_fn=activation_fn, weight=params[0], bias=params[1], trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)
                elif cnt == len(dim_layers)-1 and output_type is None:
                    layer_tmp, param_tmp = new_fc_layer(layers[cnt-1], dim_layers[cnt], activation_fn=None, weight=params[2*cnt], bias=params[2*cnt+1], trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)
                elif cnt == len(dim_layers)-1 and output_type is 'same':
                    layer_tmp, param_tmp = new_fc_layer(layers[cnt-1], dim_layers[cnt], activation_fn=activation_fn, weight=params[2*cnt], bias=params[2*cnt+1], trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)
                else:
                    layer_tmp, param_tmp = new_fc_layer(layers[cnt-1], dim_layers[cnt], activation_fn=activation_fn, weight=params[2*cnt], bias=params[2*cnt+1], trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)
                layers.append(layer_tmp)
                params_to_return = params_to_return + param_tmp
    return (layers, params_to_return)



############################################################
#####    functions for adding convolutional network    #####
############################################################
#### function to add 2D convolutional layer
def new_cnn_layer(layer_input, k_size, stride_size=[1, 1, 1, 1], activation_fn=tf.nn.relu, weight=None, bias=None, padding_type='SAME', max_pooling=False, pool_size=None, trainable=True, use_numpy_var_in_graph=False):
    with tf.name_scope('conv_layer'):
        if weight is None:
            weight = new_weight(shape=k_size, trainable=trainable)
        elif (type(weight) == np.ndarray) and not use_numpy_var_in_graph:
            weight = new_weight(shape=k_size, init_tensor=weight, trainable=trainable)
        if bias is None:
            bias = new_bias(shape=[k_size[-1]], trainable=trainable)
        elif (type(bias) == np.ndarray) and not use_numpy_var_in_graph:
            bias = new_bias(shape=[k_size[-1]], init_tensor=bias, trainable=trainable)

        conv_layer = tf.nn.conv2d(layer_input, weight, strides=stride_size, padding=padding_type) + bias

        if not (activation_fn is None):
            conv_layer = activation_fn(conv_layer)

        if max_pooling and (pool_size[1] > 1 or pool_size[2] > 1):
            layer = tf.nn.max_pool(conv_layer, ksize=pool_size, strides=pool_size, padding=padding_type)
        else:
            layer = conv_layer
    return (layer, [weight, bias])

#### function to generate network of convolutional layers
####      conv-pool-conv-pool-...-conv-pool-flat-dropout
####      k_sizes/stride_size/pool_sizes : [x_0, y_0, x_1, y_1, ..., x_m, y_m]
####      ch_sizes : [img_ch, ch_0, ch_1, ..., ch_m]
def new_cnn_net(net_input, k_sizes, ch_sizes, stride_sizes, activation_fn=tf.nn.relu, params=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, flat_output=False, trainable=True, use_numpy_var_in_graph=False):
    if not max_pool:
        pool_sizes = [None for _ in range(len(k_sizes))]

    if params is None:
        params = [None for _ in range(len(k_sizes))]

    layers, params_to_return = [], []
    with tf.name_scope('conv_net'):
        if len(k_sizes) < 2:
            #### for the case that hard-parameter shared network does not have shared layers
            layers.append(net_input)
        else:
            for layer_cnt in range(len(k_sizes)//2):
                if layer_cnt == 0:
                    layer_tmp, param_tmp = new_cnn_layer(layer_input=net_input, k_size=k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt+2], stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=activation_fn, weight=params[2*layer_cnt], bias=params[2*layer_cnt+1], padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)
                else:
                    layer_tmp, param_tmp = new_cnn_layer(layer_input=layers[layer_cnt-1], k_size=k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt+2], stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=activation_fn, weight=params[2*layer_cnt], bias=params[2*layer_cnt+1], padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)
                layers.append(layer_tmp)
                params_to_return = params_to_return + param_tmp

        #### flattening output
        if flat_output:
            output_dim = [int(layers[-1].shape[1]*layers[-1].shape[2]*layers[-1].shape[3])]
            layers.append(tf.reshape(layers[-1], [-1, output_dim[0]]))
        else:
            output_dim = layers[-1].shape[1:]

        #### add dropout layer
        if dropout:
            layers.append(tf.nn.dropout(layers[-1], dropout_prob))
    return (layers, params_to_return, output_dim)

#### function to generate network of cnn->ffnn
def new_cnn_fc_net(net_input, k_sizes, ch_sizes, stride_sizes, fc_sizes, cnn_activation_fn=tf.nn.relu, cnn_params=None, fc_activation_fn=tf.nn.relu, fc_params=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, output_type=None, trainable=True, use_numpy_var_in_graph=False):
    cnn_model, cnn_params, cnn_output_dim = new_cnn_net(net_input, k_sizes, ch_sizes, stride_sizes, activation_fn=cnn_activation_fn, params=cnn_params, padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, dropout=dropout, dropout_prob=dropout_prob, flat_output=True, trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)

    fc_model, fc_params = new_fc_net(cnn_model[-1], fc_sizes, activation_fn=fc_activation_fn, params=fc_params, output_type=output_type, trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)
    return (cnn_model+fc_model, cnn_params, fc_params)



##############################################################################
################# Hard-Parameter Sharing Model
##############################################################################
#### function to generate HPS model of CNN-FC ver. (hard-shared conv layers -> task-dependent fc layers)
def new_hardparam_cnn_fc_nets(net_inputs, k_sizes, ch_sizes, stride_sizes, fc_sizes, num_task, cnn_activation_fn=tf.nn.relu, cnn_params=None, fc_activation_fn=tf.nn.relu, fc_params=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, output_type=None):
    num_acc_specific_params, num_specific_params_tmp = [0], 0
    for a in fc_sizes:
        num_specific_params_tmp += 2 * len(a)
        num_acc_specific_params.append(num_specific_params_tmp)

    models, cnn_params_return, fc_params_return = [], [], []
    for task_cnt in range(num_task):
        if task_cnt == 0 and fc_params is None:
            model_tmp, cnn_params_return, fc_param_tmp = new_cnn_fc_net(net_inputs[task_cnt], k_sizes, ch_sizes, stride_sizes, fc_sizes[task_cnt], cnn_activation_fn=cnn_activation_fn, cnn_params=cnn_params, fc_activation_fn=fc_activation_fn, fc_params=None, padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, dropout=dropout, dropout_prob=dropout_prob, output_type=output_type)
        elif task_cnt == 0:
            model_tmp, cnn_params_return, fc_param_tmp = new_cnn_fc_net(net_inputs[task_cnt], k_sizes, ch_sizes, stride_sizes, fc_sizes[task_cnt], cnn_activation_fn=cnn_activation_fn, cnn_params=cnn_params, fc_activation_fn=fc_activation_fn, fc_params=fc_params[task_cnt], padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, dropout=dropout, dropout_prob=dropout_prob, output_type=output_type)
        elif fc_params is None:
            model_tmp, _, fc_param_tmp = new_cnn_fc_net(net_inputs[task_cnt], k_sizes, ch_sizes, stride_sizes, fc_sizes[task_cnt], cnn_activation_fn=cnn_activation_fn, cnn_params=cnn_params_return, fc_activation_fn=fc_activation_fn, fc_params=None, padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, dropout=dropout, dropout_prob=dropout_prob, output_type=output_type)
        else:
            model_tmp, _, fc_param_tmp = new_cnn_fc_net(net_inputs[task_cnt], k_sizes, ch_sizes, stride_sizes, fc_sizes[task_cnt], cnn_activation_fn=cnn_activation_fn, cnn_params=cnn_params_return, fc_activation_fn=fc_activation_fn, fc_params=fc_params[task_cnt], padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, dropout=dropout, dropout_prob=dropout_prob, output_type=output_type)

        models.append(model_tmp)
        fc_params_return.append(fc_param_tmp)

    return (models, cnn_params_return, fc_params_return)



##############################################################################
################# Tensor Factored Model
##############################################################################
#### function to generate parameters of tensor factored convolutional layer
def new_tensorfactored_weight(shape, num_task, factor_type='Tucker', factor_eps_or_k=0.01, init_val=None):
    if init_val is None:
        if len(shape) == 2:
            W_init = np.random.rand(shape[0], shape[1], num_task)
        elif len(shape) == 4:
            W_init = np.random.rand(shape[0], shape[1], shape[2], shape[3], num_task)
        else:
            return (None, None)
    else:
        W_init = init_val

    W_tmp, W_dict = TensorProducer(W_init, factor_type, eps_or_k=factor_eps_or_k, return_true_var=True)

    if len(shape) == 2:
        W = [W_tmp[:, :, i] for i in range(num_task)]
    elif len(shape) == 4:
        W = [W_tmp[:, :, :, :, i] for i in range(num_task)]
    return (W, W_dict)

def new_tensorfactored_cnn_weights(k_sizes, ch_sizes, num_task, factor_type='Tucker', factor_eps_or_k=0.01, init_params=None):
    num_layers = len(ch_sizes)-1
    if init_params is None:
        init_params = [None for _ in range(num_layers)]

    param_tmp = [[] for i in range(num_task)]
    for layer_cnt in range(num_layers):
        W_tmp, _ = new_tensorfactored_weight(k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt+2], num_task, factor_type, factor_eps_or_k, init_params[layer_cnt])
        bias_tmp = [new_bias(shape=[ch_sizes[layer_cnt+1]]) for i in range(num_task)]
        for task_cnt in range(num_task):
            param_tmp[task_cnt].append(W_tmp[task_cnt])
            param_tmp[task_cnt].append(bias_tmp[task_cnt])

    param = []
    for task_cnt in range(num_task):
        param = param + param_tmp[task_cnt]
    return param


def new_tensorfactored_cnn_nets(net_inputs, k_sizes, ch_sizes, stride_sizes, num_task, activation_fn=tf.nn.relu, params=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, flat_output=False, factor_type='Tucker', factor_eps_or_k=0.01, init_params=None):
    num_para_per_model = len(k_sizes)

    with tf.name_scope('TF_conv_net'):
        if len(k_sizes) < 2:
            #### for the case that hard-parameter shared network does not have shared layers
            return (net_inputs, [])
        elif params is None:
            #### network & parameters are new
            params = new_tensorfactored_cnn_weights(k_sizes, ch_sizes, num_task, factor_type, factor_eps_or_k, init_params)

            # params
            cnn_models = []
            for task_cnt in range(num_task):
                cnn_model_tmp, _, output_dim = new_cnn_net(net_inputs[task_cnt], k_sizes, ch_sizes, stride_sizes, activation_fn=activation_fn, params=params[task_cnt*num_para_per_model:(task_cnt+1)*num_para_per_model], padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, dropout=dropout, dropout_prob=dropout_prob, flat_output=flat_output)
                cnn_models.append(cnn_model_tmp)
        else:
            #### network generated from existing parameters
            cnn_models = []
            for task_cnt in range(num_task):
                cnn_model_tmp, _, output_dim = new_cnn_net(net_inputs[task_cnt], k_sizes, ch_sizes, stride_sizes, activation_fn=activation_fn, params=params[task_cnt*num_para_per_model:(task_cnt+1)*num_para_per_model], padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, dropout=dropout, dropout_prob=dropout_prob, flat_output=flat_output)
                cnn_models.append(cnn_model_tmp)
    return (cnn_models, params, output_dim)

def new_tensorfactored_cnn_fc_nets(net_inputs, k_sizes, ch_sizes, stride_sizes, fc_sizes, num_task, cnn_activation_fn=tf.nn.relu, cnn_params=None, fc_activation_fn=tf.nn.relu, fc_params=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, factor_type='Tucker', factor_eps_or_k=0.01, output_type=None, init_param=None):
    num_acc_specific_params, num_specific_params_tmp = [0], 0
    for a in fc_sizes:
        num_specific_params_tmp += 2 * len(a)
        num_acc_specific_params.append(num_specific_params_tmp)

    num_cnn_layers = len(k_sizes)//2
    assert ((init_param is None) or ( (init_param is not None) and (len(init_param)==2*num_cnn_layers*num_task+num_acc_specific_params[-1]) )), "Given initializing parameter doesn't match to the size of architecture"
    if init_param is None:
        layerwise_cnn_init_params = [None for _ in range(num_cnn_layers)]
    else:
        layerwise_cnn_init_params = []
        for cnn_layer_cnt in range(num_cnn_layers):
            layerwise_cnn_param_tmp = []
            for task_cnt in range(num_task):
                layerwise_cnn_param_tmp.append(init_param[2*(cnn_layer_cnt+num_cnn_layers*task_cnt)+num_acc_specific_params[task_cnt]])
            layerwise_cnn_param = np.transpose(np.stack(layerwise_cnn_param_tmp), axes=[1, 2, 3, 4, 0])
            layerwise_cnn_init_params.append(layerwise_cnn_param)

    if cnn_params is None:
        cnn_models, cnn_params, output_dim = new_tensorfactored_cnn_nets(net_inputs, k_sizes, ch_sizes, stride_sizes, num_task, activation_fn=cnn_activation_fn, params=cnn_params, padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, dropout=dropout, dropout_prob=dropout_prob, flat_output=True, factor_type=factor_type, factor_eps_or_k=factor_eps_or_k, init_params=layerwise_cnn_init_params)
    else:
        cnn_models, _, output_dim = new_tensorfactored_cnn_nets(net_inputs, k_sizes, ch_sizes, stride_sizes, num_task, activation_fn=cnn_activation_fn, params=cnn_params, padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, dropout=dropout, dropout_prob=dropout_prob, flat_output=True, factor_type=factor_type, factor_eps_or_k=factor_eps_or_k)

    fc_models = []
    if fc_params is None:
        fc_params = []
        for task_cnt in range(num_task):
            fc_model_tmp, fc_param_tmp = new_fc_net(cnn_models[task_cnt][-1], fc_sizes[task_cnt], activation_fn=fc_activation_fn, params=None, output_type=output_type)
            fc_models.append(fc_model_tmp)
            fc_params = fc_params + fc_param_tmp
    else:
        for task_cnt in range(num_task):
            fc_model_tmp, _ = new_fc_net(cnn_models[task_cnt][-1], fc_sizes[task_cnt], activation_fn=fc_activation_fn, params=fc_params[num_acc_specific_params[task_cnt]:num_acc_specific_params[task_cnt+1]], output_type=output_type)
            fc_models.append(fc_model_tmp)

    models = []
    for task_cnt in range(num_task):
        models.append(cnn_models[task_cnt]+fc_models[task_cnt])
    return (models, cnn_params, fc_params)



##############################################################################
################# Progressive Neural Net model
##############################################################################

#### function to add progressive convolution layer
# k_size : [h, w, cin, cout]
def new_progressive_cnn_layer(layer_input, k_size, stride_size=[1, 1, 1, 1], activation_fn=tf.nn.relu, weight=None, bias=None, padding_type='SAME', max_pooling=False, pool_size=None, prev_column_inputs=None, num_prev_cols=1, lat_connect_param=None, trainable=True, dim_reduction_scale=1.0):
    with tf.name_scope('prog_conv_layer'):
        if weight is None:
            weight = new_weight(shape=k_size, trainable=trainable)
        if bias is None:
            bias = new_bias(shape=[k_size[-1]], trainable=trainable)

        conv_layer = tf.nn.conv2d(layer_input, weight, strides=stride_size, padding=padding_type) + bias

        exist_prev_column = (None not in prev_column_inputs) if (type(prev_column_inputs) == list) else (prev_column_inputs is not None)

        if exist_prev_column:
            exist_lat_param = (not (all([x is None for x in lat_connect_param]))) if (type(lat_connect_param) == list) else (lat_connect_param is not None)
            if not exist_lat_param:
                ## generate new params for lateral connection (alpha, (Wv, bv), Wu)
                lat_connect_param, nic = [], int(k_size[2] / (dim_reduction_scale * (num_prev_cols+1.0)))
                for col_cnt in range(num_prev_cols):
                    lat_connect_param = lat_connect_param + [new_weight(shape=[1], trainable=trainable), new_weight(shape=[1, 1, k_size[2], nic], trainable=trainable), new_bias(shape=[nic], trainable=trainable), new_weight(shape=k_size[0:2]+[nic, k_size[3]], trainable=trainable)]

            ## generate lateral connection
            lateral_outputs = []
            for col_cnt in range(num_prev_cols):
                lat_col_hid1 = tf.multiply(lat_connect_param[4*col_cnt], prev_column_inputs[col_cnt])

                ## dim reduction using conv (k:[1, 1], stride:[1, 1], act:ReLu)
                lat_col_hid2 = tf.nn.relu(tf.nn.conv2d(lat_col_hid1, lat_connect_param[4*col_cnt+1], strides=[1, 1, 1, 1], padding=padding_type) + lat_connect_param[4*col_cnt+2])

                ## conv lateral connection
                lateral_outputs.append(tf.nn.conv2d(lat_col_hid2, lat_connect_param[4*col_cnt+3], strides=stride_size, padding=padding_type))

            conv_layer = conv_layer + tf.reduce_sum(lateral_outputs, axis=0)
        else:
            lat_connect_param = [None for _ in range(4*num_prev_cols)]

        if activation_fn is not None:
            act_conv_layer = activation_fn(conv_layer)

        if max_pooling:
            layer = tf.nn.max_pool(act_conv_layer, ksize=pool_size, strides=pool_size, padding=padding_type)
        else:
            layer = act_conv_layer
    return (layer, [weight, bias], lat_connect_param)


#### function to generate network of progressive convolutional layers
####      prog_conv-pool-prog_conv-pool-...-prog_conv-pool-flat-dropout
####      k_sizes/stride_size/pool_sizes : [x_0, y_0, x_1, y_1, ..., x_m, y_m]
####      ch_sizes : [img_ch, ch_0, ch_1, ..., ch_m]
def new_progressive_cnn_net(net_input, k_sizes, ch_sizes, stride_sizes, activation_fn=tf.nn.relu, params=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, flat_output=False, prev_column_net=None, lat_connect_params=None, trainable=True, dim_reduction_scale=1.0):
    num_layers = len(k_sizes)//2
    assert (num_layers == len(ch_sizes)-1), "Check the number of progressive cnn layers"

    if not max_pool:
        pool_sizes = [None for _ in range(len(k_sizes))]

    if prev_column_net is None:
        prev_column_net = [[None for _ in range(num_layers)]]
    num_prev_nets = len(prev_column_net)
    lat_param_cnt_multiplier = 4*num_prev_nets

    if lat_connect_params is None:
        lat_connect_params = [None for _ in range(lat_param_cnt_multiplier*num_layers)]

    layers_for_skip, next_skip_connect = [net_input], None
    with tf.name_scope('prog_conv_net'):
        if num_layers < 1:
            #### for the case that hard-parameter shared network does not have shared layers
            return (net_input, [])
        elif params is None:
            #### network & parameters are new
            layers, params, lat_params = [], [], []
            for layer_cnt in range(num_layers):
                if layer_cnt == 0:
                    layer_tmp, para_tmp, lat_para_tmp = new_progressive_cnn_layer(layer_input=net_input, k_size=k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt+2], stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=activation_fn, padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], num_prev_cols=num_prev_nets, trainable=trainable, dim_reduction_scale=dim_reduction_scale)
                else:
                    layer_tmp, para_tmp, lat_para_tmp = new_progressive_cnn_layer(layer_input=layers[layer_cnt-1], k_size=k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt+2], stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=activation_fn, padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], prev_column_inputs=[prev_column_net[c][layer_cnt-1] for c in range(num_prev_nets)], num_prev_cols=num_prev_nets, lat_connect_param=lat_connect_params[lat_param_cnt_multiplier*layer_cnt:lat_param_cnt_multiplier*(layer_cnt+1)], trainable=trainable, dim_reduction_scale=dim_reduction_scale)
                layers.append(layer_tmp)
                layers_for_skip.append(layer_tmp)
                params = params + para_tmp
                lat_params = lat_params + lat_para_tmp
        else:
            #### network generated from existing parameters
            layers, lat_params = [], lat_connect_params
            for layer_cnt in range(num_layers):
                if layer_cnt == 0:
                    layer_tmp, _, _ = new_progressive_cnn_layer(layer_input=net_input, k_size=k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt+2], stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=activation_fn, weight=params[2*layer_cnt], bias=params[2*layer_cnt+1], padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], num_prev_cols=num_prev_nets, trainable=trainable, dim_reduction_scale=dim_reduction_scale)
                else:
                    layer_tmp, _, _ = new_progressive_cnn_layer(layer_input=layers[layer_cnt-1], k_size=k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt+2], stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=activation_fn, weight=params[2*layer_cnt], bias=params[2*layer_cnt+1], padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], prev_column_inputs=[prev_column_net[c][layer_cnt-1] for c in range(num_prev_nets)], num_prev_cols=num_prev_nets, lat_connect_param=lat_connect_params[lat_param_cnt_multiplier*layer_cnt:lat_param_cnt_multiplier*(layer_cnt+1)], trainable=trainable, dim_reduction_scale=dim_reduction_scale)
                layers.append(layer_tmp)
                layers_for_skip.append(layer_tmp)

        #### flattening output
        if flat_output:
            output_dim = [int(layers[-1].shape[1]*layers[-1].shape[2]*layers[-1].shape[3])]
            layers.append(tf.reshape(layers[-1], [-1, output_dim[0]]))
        else:
            output_dim = layers[-1].shape[1:]

        #### add dropout layer
        if dropout:
            layers.append(tf.nn.dropout(layers[-1], dropout_prob))
    return (layers, params, lat_params, output_dim)


#### function to generate network of progressive cnn->ffnn
def new_progressive_cnn_fc_net(net_input, k_sizes, ch_sizes, stride_sizes, fc_sizes, cnn_activation_fn=tf.nn.relu, cnn_params=None, fc_activation_fn=tf.nn.relu, fc_params=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, output_type=None, prev_net=None, cnn_lateral_params=None, trainable=True, dim_reduction_scale=1.0, use_numpy_var_in_graph=False):
    cnn_model, cnn_params, cnn_lateral_params, cnn_output_dim = new_progressive_cnn_net(net_input, k_sizes, ch_sizes, stride_sizes, activation_fn=cnn_activation_fn, params=cnn_params, padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, dropout=dropout, dropout_prob=dropout_prob, flat_output=True, prev_column_net=prev_net, lat_connect_params=cnn_lateral_params, trainable=trainable, dim_reduction_scale=dim_reduction_scale)

    fc_model, fc_params = new_fc_net(cnn_model[-1], fc_sizes, activation_fn=fc_activation_fn, params=fc_params, output_type=output_type, use_numpy_var_in_graph=use_numpy_var_in_graph)
    return (cnn_model+fc_model, cnn_params, cnn_lateral_params, fc_params)
