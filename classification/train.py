import os, sys
import timeit
from random import shuffle

import numpy as np
import tensorflow as tf
from scipy.io import savemat

from utils.utils import savemat_wrapper
if sys.version_info.major < 3:
    from gen_data import mnist_data_print_info, cifar_data_print_info, officehome_data_print_info
else:
    from classification.gen_data import mnist_data_print_info, cifar_data_print_info, officehome_data_print_info

from classification.model.cnn_baseline_model import MTL_several_CNN_minibatch, MTL_CNN_minibatch, MTL_CNN_HPS_minibatch, MTL_CNN_tensorfactor_minibatch, MTL_CNN_progressive_minibatch
from classification.model.cnn_den_model import CNN_FC_DEN
from classification.model.cnn_df_model import Deconvolutional_Factorized_CNN, Deconvolutional_Factorized_CNN_Direct, Deconvolutional_Factorized_CNN_tc2

#### function to generate appropriate deep neural network
def model_generation(model_architecture, model_hyperpara, train_hyperpara, data_info, tfInitParam=None):
    learning_model, gen_model_success = None, True
    learning_rate = train_hyperpara['lr']
    learning_rate_decay = train_hyperpara['lr_decay']

    if len(data_info) == 3:
        x_dim, y_dim, y_depth = data_info
    elif len(data_info) == 4:
        x_dim, y_dim, y_depth, num_task = data_info

    if isinstance(y_depth, list) or type(y_depth) == np.ndarray:
        fc_hidden_sizes = [list(model_hyperpara['hidden_layer'])+[y_d] for y_d in y_depth]
    else:
        fc_hidden_size = model_hyperpara['hidden_layer'] + [y_depth]
        fc_hidden_sizes = [fc_hidden_size for _ in range(num_task)]

    if 'batch_size' in model_hyperpara:
        batch_size = model_hyperpara['batch_size']
    input_size = model_hyperpara['image_dimension']

    cnn_kernel_size, cnn_kernel_stride, cnn_channel_size = model_hyperpara['kernel_sizes'], model_hyperpara['stride_sizes'], model_hyperpara['channel_sizes']
    cnn_padding, cnn_pooling, cnn_dropout = model_hyperpara['padding_type'], model_hyperpara['max_pooling'], model_hyperpara['dropout']
    if cnn_pooling:
        cnn_pool_size = model_hyperpara['pooling_size']
    else:
        cnn_pool_size = None

    if 'deconvolutional_factorized' in model_architecture:
        cnn_know_base_size, cnn_task_specific_size, cnn_deconv_stride_size = model_hyperpara['cnn_KB_sizes'], model_hyperpara['cnn_TS_sizes'], model_hyperpara['cnn_deconv_stride_sizes']

    ###### CNN models
    if model_architecture == 'mtl_several_cnn_minibatch':
        print("Training STL-CNNs model (one independent NN per task)")
        learning_model = MTL_several_CNN_minibatch(dim_channels=cnn_channel_size, num_tasks=num_task, dim_fcs=fc_hidden_size, input_size=input_size, dim_kernel=cnn_kernel_size, dim_strides=cnn_kernel_stride, batch_size=batch_size, learning_rate=learning_rate, learning_rate_decay=learning_rate_decay, padding_type=cnn_padding, max_pooling=cnn_pooling, dim_pool=cnn_pool_size, dropout=cnn_dropout)
    elif model_architecture == 'mtl_cnn_minibatch':
        print("Training MTL-CNN model (Single NN for all tasks)")
        learning_model = MTL_CNN_minibatch(dim_channels=cnn_channel_size, num_tasks=num_task, dim_fcs=fc_hidden_size, input_size=input_size, dim_kernel=cnn_kernel_size, dim_strides=cnn_kernel_stride, batch_size=batch_size, learning_rate=learning_rate, learning_rate_decay=learning_rate_decay, padding_type=cnn_padding, max_pooling=cnn_pooling, dim_pool=cnn_pool_size, dropout=cnn_dropout)
    elif 'mtl_cnn_hps' in model_architecture:
        print("Training MTL-CNN model (Hard-Para Sharing ver.)")
        learning_model = MTL_CNN_HPS_minibatch(dim_channels=cnn_channel_size, num_tasks=num_task, dim_fcs=fc_hidden_sizes, input_size=input_size, dim_kernel=cnn_kernel_size, dim_strides=cnn_kernel_stride, batch_size=batch_size, learning_rate=learning_rate, learning_rate_decay=learning_rate_decay, padding_type=cnn_padding, max_pooling=cnn_pooling, dim_pool=cnn_pool_size, dropout=cnn_dropout)
    elif 'mtl_cnn_tensorfactor' in model_architecture:
        print("Training MTL-CNN model (Tensorfactorization ver.)")
        factor_type = model_hyperpara['tensor_factor_type']
        factor_eps_or_k = model_hyperpara['tensor_factor_error_threshold']
        learning_model = MTL_CNN_tensorfactor_minibatch(dim_channels=cnn_channel_size, num_tasks=num_task, dim_fcs=fc_hidden_sizes, input_size=input_size, dim_kernel=cnn_kernel_size, dim_strides=cnn_kernel_stride, batch_size=batch_size, learning_rate=learning_rate, learning_rate_decay=learning_rate_decay, padding_type=cnn_padding, max_pooling=cnn_pooling, dim_pool=cnn_pool_size, dropout=cnn_dropout, factor_type=factor_type, factor_eps_or_k=factor_eps_or_k, init_param=tfInitParam)
    elif 'cnn_progressive' in model_architecture:
        print("Training LL-CNN Progressive model")
        dim_red_scale = model_hyperpara['dim_reduction_scale']
        learning_model = MTL_CNN_progressive_minibatch(dim_channels=cnn_channel_size, dim_fcs=fc_hidden_size, input_size=input_size, dim_kernel=cnn_kernel_size, dim_strides=cnn_kernel_stride, batch_size=batch_size, learning_rate=learning_rate, learning_rate_decay=learning_rate_decay, padding_type=cnn_padding, max_pooling=cnn_pooling, dim_pool=cnn_pool_size, dropout=cnn_dropout, dim_reduction_scale=dim_red_scale)
    elif ('cnn_den' in model_architecture or 'cnn_dynamically' in model_architecture):
        print("Training LL-CNN Dynamically Expandable model")
        learning_model = CNN_FC_DEN(model_hyperpara, train_hyperpara, data_info)

    elif model_architecture == 'deconvolutional_factorized_cnn':
        print("Training DF-CNN model (IJCAI 2019)")
        learning_model = Deconvolutional_Factorized_CNN(num_task, cnn_channel_size, fc_hidden_size, input_size, cnn_kernel_size, cnn_kernel_stride, cnn_know_base_size, cnn_task_specific_size, cnn_deconv_stride_size, batch_size, learning_rate, learning_rate_decay, padding_type=cnn_padding, max_pooling=cnn_pooling, dim_pool=cnn_pool_size, dropout=cnn_dropout, relation_activation_fn_cnn=None)
    elif model_architecture == 'deconvolutional_factorized_cnn_direct':
        print("Training ablated DF-CNN model (DF-CNN.direct)")
        learning_model = Deconvolutional_Factorized_CNN_Direct(num_task, cnn_channel_size, fc_hidden_size, input_size, cnn_kernel_size, cnn_kernel_stride, cnn_know_base_size, cnn_task_specific_size, cnn_deconv_stride_size, batch_size, learning_rate, learning_rate_decay, padding_type=cnn_padding, max_pooling=cnn_pooling, dim_pool=cnn_pool_size, dropout=cnn_dropout, relation_activation_fn_cnn=None)
    elif model_architecture == 'deconvolutional_factorized_cnn_tc2':
        print("Training ablated DF-CNN model (DF-CNN.tc2)")
        learning_model = Deconvolutional_Factorized_CNN_tc2(num_task, cnn_channel_size, fc_hidden_size, input_size, cnn_kernel_size, cnn_kernel_stride, cnn_know_base_size, cnn_task_specific_size, cnn_deconv_stride_size, batch_size, learning_rate, learning_rate_decay, padding_type=cnn_padding, max_pooling=cnn_pooling, dim_pool=cnn_pool_size, dropout=cnn_dropout, relation_activation_fn_cnn=None)

    else:
        print("No such model exists!!")
        print("No such model exists!!")
        print("No such model exists!!")
        gen_model_success = False
    return (learning_model, gen_model_success)


#### module of training/testing one model
def train(model_architecture, model_hyperpara, train_hyperpara, dataset, data_type, doLifelong, useGPU=False, GPU_device=0, save_param=False, param_folder_path='saved_param', save_param_interval=100, save_graph=False, tfInitParam=None):
    assert ('progressive' not in model_architecture and 'den' not in model_architecture and 'dynamically' not in model_architecture), "Use train function appropriate to the architecture"

    config = tf.ConfigProto()
    if useGPU:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU_device)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.97
        print("GPU %d is used" %(GPU_device))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]=""
        print("CPU is used")

    ## This order of tasks for training can be arbitrary.
    task_training_order = list(range(train_hyperpara['num_tasks']))
    task_for_train, task_change_epoch = task_training_order.pop(0), [1]

    ### set-up data
    train_data, validation_data, test_data = dataset
    if 'mnist' in data_type:
        num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth = mnist_data_print_info(train_data, validation_data, test_data, True, print_info=False)
    elif ('cifar10' in data_type) and not ('cifar100' in data_type):
        num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth = cifar_data_print_info(train_data, validation_data, test_data, True, print_info=False)
    elif 'cifar100' in data_type:
        num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth = cifar_data_print_info(train_data, validation_data, test_data, True, print_info=False)
    elif 'officehome' in data_type:
        num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth = officehome_data_print_info(train_data, validation_data, test_data, True, print_info=False)

    ### Set hyperparameter related to training process
    learning_step_max = train_hyperpara['learning_step_max']
    improvement_threshold = train_hyperpara['improvement_threshold']
    patience = train_hyperpara['patience']
    patience_multiplier = train_hyperpara['patience_multiplier']
    batch_size = model_hyperpara['batch_size']

    ### Generate Model
    learning_model, generation_success = model_generation(model_architecture, model_hyperpara, train_hyperpara, [x_dim, y_dim, y_depth, num_task], tfInitParam=tfInitParam)
    if not generation_success:
        return (None, None, None, None)

    ### Training Procedure
    best_param = []
    if save_param:
        best_para_file_name = param_folder_path+'/best_model_parameter'
        print("Saving trained parameters at '%s'" %(param_folder_path) )
    else:
        print("Not saving trained parameters")

    learning_step = -1
    if (('batch_size' in locals()) or ('batch_size' in globals())) and (('num_task' in locals()) or ('num_task' in globals())):
        if num_task > 1:
            indices = [list(range(num_train[x])) for x in range(num_task)]
        else:
            indices = [list(range(num_train[0]))]

    best_valid_accuracy, test_accuracy_at_best_epoch, best_epoch, epoch_bias = 0.0, 0.0, -1, 0
    train_accuracy_hist, valid_accuracy_hist, test_accuracy_hist, best_test_accuracy_hist = [], [], [], []

    model_train_acc, model_valid_acc, model_test_acc = learning_model.train_accuracy, learning_model.valid_accuracy, learning_model.test_accuracy
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if save_graph:
            tfboard_writer = tf.summary.FileWriter('./graphs', sess.graph)

        start_time = timeit.default_timer()
        while learning_step < min(learning_step_max, epoch_bias + patience):
            learning_step = learning_step+1

            if not doLifelong:
                task_for_train = np.random.randint(0, num_task)

            #### training process
            if learning_step > 0:
                shuffle(indices[task_for_train])

                for batch_cnt in range(num_train[task_for_train]//batch_size):
                    batch_train_x = train_data[task_for_train][0][indices[task_for_train][batch_cnt*batch_size:(batch_cnt+1)*batch_size], :]
                    batch_train_y = train_data[task_for_train][1][indices[task_for_train][batch_cnt*batch_size:(batch_cnt+1)*batch_size]]
                    sess.run(learning_model.update[task_for_train], feed_dict={learning_model.model_input[task_for_train]: batch_train_x, learning_model.true_output[task_for_train]: batch_train_y, learning_model.epoch: learning_step-1, learning_model.dropout_prob: 0.5})

            #### evaluation process
            train_accuracy_tmp = [0.0 for _ in range(num_task)]
            validation_accuracy_tmp = [0.0 for _ in range(num_task)]
            test_accuracy_tmp = [0.0 for _ in range(num_task)]
            for task_cnt in range(num_task):
                for batch_cnt in range(num_train[task_cnt]//batch_size):
                    train_accuracy_tmp[task_cnt] = train_accuracy_tmp[task_cnt] + sess.run(model_train_acc[task_cnt], feed_dict={learning_model.model_input[task_cnt]: train_data[task_cnt][0][batch_cnt*batch_size:(batch_cnt+1)*batch_size, :], learning_model.true_output[task_cnt]: train_data[task_cnt][1][batch_cnt*batch_size:(batch_cnt+1)*batch_size], learning_model.dropout_prob: 1.0})
                train_accuracy_tmp[task_cnt] = train_accuracy_tmp[task_cnt]/((num_train[task_cnt]//batch_size)*batch_size)

                for batch_cnt in range(num_valid[task_cnt]//batch_size):
                    validation_accuracy_tmp[task_cnt] = validation_accuracy_tmp[task_cnt] + sess.run(model_valid_acc[task_cnt], feed_dict={learning_model.model_input[task_cnt]: validation_data[task_cnt][0][batch_cnt*batch_size:(batch_cnt+1)*batch_size, :], learning_model.true_output[task_cnt]: validation_data[task_cnt][1][batch_cnt*batch_size:(batch_cnt+1)*batch_size], learning_model.dropout_prob: 1.0})
                validation_accuracy_tmp[task_cnt] = validation_accuracy_tmp[task_cnt]/((num_valid[task_cnt]//batch_size)*batch_size)

                for batch_cnt in range(num_test[task_cnt]//batch_size):
                    test_accuracy_tmp[task_cnt] = test_accuracy_tmp[task_cnt] + sess.run(model_test_acc[task_cnt], feed_dict={learning_model.model_input[task_cnt]: test_data[task_cnt][0][batch_cnt*batch_size:(batch_cnt+1)*batch_size, :], learning_model.true_output[task_cnt]: test_data[task_cnt][1][batch_cnt*batch_size:(batch_cnt+1)*batch_size], learning_model.dropout_prob: 1.0})
                test_accuracy_tmp[task_cnt] = test_accuracy_tmp[task_cnt]/((num_test[task_cnt]//batch_size)*batch_size)

            train_accuracy, valid_accuracy, test_accuracy = sum(train_accuracy_tmp)/num_task, sum(validation_accuracy_tmp)/num_task, sum(test_accuracy_tmp)/num_task
            if doLifelong:
                train_accuracy_to_compare, valid_accuracy_to_compare, test_accuracy_to_compare = train_accuracy_tmp[task_for_train], validation_accuracy_tmp[task_for_train], test_accuracy_tmp[task_for_train]
            else:
                train_accuracy_to_compare, valid_accuracy_to_compare, test_accuracy_to_compare = train_accuracy, valid_accuracy, test_accuracy
            print('epoch %d - Train : %f, Validation : %f' % (learning_step, abs(train_accuracy_to_compare), abs(valid_accuracy_to_compare)))

            #### when validation accuracy is better than before
            if valid_accuracy_to_compare > best_valid_accuracy:
                str_temp = ''
                if valid_accuracy_to_compare > best_valid_accuracy * improvement_threshold:
                    ## for early-stopping
                    patience = max(patience, (learning_step-epoch_bias)*patience_multiplier)
                    str_temp = '\t<<'
                best_valid_accuracy, best_epoch = valid_accuracy_to_compare, learning_step
                test_accuracy_at_best_epoch = test_accuracy_to_compare
                print('\t\t\t\t\t\t\tTest : %f%s' % (abs(test_accuracy_at_best_epoch), str_temp))

            #### switch to new task (only for lifelong learning)
            if doLifelong and learning_step >= epoch_bias+min(patience, learning_step_max//num_task) and len(task_training_order) > 0:
                print('\n\t>>Change to new task!<<\n')

                if save_param:
                    para_file_name = param_folder_path + '/model_parameter_t%d.mat'%(task_for_train)
                    curr_param = learning_model.get_params_val(sess)
                    savemat(para_file_name, {'parameter': curr_param})

                # update epoch_bias, task_for_train, task_change_epoch
                epoch_bias, task_for_train = learning_step, task_training_order.pop(0)
                task_change_epoch.append(learning_step+1)

                # initialize best_valid_accuracy, best_epoch, patience
                patience = train_hyperpara['patience']
                best_valid_accuracy, best_epoch = 0.0, -1

            train_accuracy_hist.append(train_accuracy_tmp + [train_accuracy])
            valid_accuracy_hist.append(validation_accuracy_tmp + [valid_accuracy])
            test_accuracy_hist.append(test_accuracy_tmp + [test_accuracy])
            best_test_accuracy_hist.append(test_accuracy_at_best_epoch)

        if save_param:
            para_file_name = param_folder_path + '/final_model_parameter.mat'
            curr_param = learning_model.get_params_val(sess)
            savemat(para_file_name, {'parameter': curr_param})

    end_time = timeit.default_timer()
    print("End of Training")
    print("Time consumption for training : %.2f" %(end_time-start_time))
    if not doLifelong:
        print("Best validation accuracy : %.4f (at epoch %d)" %(abs(best_valid_accuracy), best_epoch))
        print("Test accuracy at that epoch (%d) : %.4f" %(best_epoch, abs(test_accuracy_at_best_epoch)))

    ## Summary of statistics during training and evaluation
    result_summary = {}
    result_summary['training_time'] = end_time - start_time
    result_summary['num_epoch'] = learning_step
    result_summary['best_epoch'] = best_epoch
    result_summary['history_train_accuracy'] = train_accuracy_hist
    result_summary['history_validation_accuracy'] = valid_accuracy_hist
    result_summary['history_test_accuracy'] = test_accuracy_hist
    result_summary['history_best_test_accuracy'] = best_test_accuracy_hist
    result_summary['best_validation_accuracy'] = abs(best_valid_accuracy)
    result_summary['test_accuracy_at_best_epoch'] = abs(test_accuracy_at_best_epoch)
    if doLifelong:
        tmp_valid_accuracy_hist = np.array(valid_accuracy_hist)
        chk_epoch = [(task_change_epoch[x], task_change_epoch[x+1]) for x in range(len(task_change_epoch)-1)] + [(task_change_epoch[-1], learning_step+1)]
        tmp_best_valid_accuracy_list = [np.amax(tmp_valid_accuracy_hist[x[0]:x[1], t]) for x, t in zip(chk_epoch, range(num_task))]
        result_summary['best_validation_accuracy'] = sum(tmp_best_valid_accuracy_list) / float(len(tmp_best_valid_accuracy_list))
        result_summary['task_changed_epoch'] = task_change_epoch

    if save_graph:
        tfboard_writer.close()

    return result_summary, learning_model.num_trainable_var



#### module of training/testing one model
def train_progressive_net(model_architecture, model_hyperpara, train_hyperpara, dataset, data_type, doLifelong, useGPU=False, GPU_device=0, save_param=False, param_folder_path='saved_param', save_param_interval=100, save_graph=False):
    assert ('progressive' in model_architecture and doLifelong), "Use train function appropriate to the architecture (Progressive Neural Net)"
    print("\nTrain function for Progressive Net is called!\n")

    config = tf.ConfigProto()
    if useGPU:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU_device)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.90
        print("GPU %d is used" %(GPU_device))
    else:
        print("CPU is used")

    ## This order of tasks for training can be arbitrary.
    task_training_order = list(range(train_hyperpara['num_tasks']))
    task_for_train, task_change_epoch = task_training_order.pop(0), [1]

    ### set-up data
    train_data, validation_data, test_data = dataset
    if 'mnist' in data_type:
        num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth = mnist_data_print_info(train_data, validation_data, test_data, True, print_info=False)
    elif 'cifar10' in data_type:
        num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth = cifar_data_print_info(train_data, validation_data, test_data, True, print_info=False)
    elif data_type == 'cifar100':
        num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth = cifar_data_print_info(train_data, validation_data, test_data, True, print_info=False)
    elif 'officehome' in data_type:
        num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth = officehome_data_print_info(train_data, validation_data, test_data, True, print_info=False)


    ### Set hyperparameter related to training process
    learning_step_max = train_hyperpara['learning_step_max']
    improvement_threshold = train_hyperpara['improvement_threshold']
    patience = train_hyperpara['patience']
    patience_multiplier = train_hyperpara['patience_multiplier']
    if 'batch_size' in model_hyperpara:
        batch_size = model_hyperpara['batch_size']


    ### Generate Model
    learning_model, generation_success = model_generation(model_architecture, model_hyperpara, train_hyperpara, [x_dim, y_dim, y_depth, num_task])
    if not generation_success:
        return (None, None, None, None)

    ### Training Procedure
    if save_param:
        best_para_file_name = param_folder_path+'/best_model_parameter'

    learning_step = -1
    if (('batch_size' in locals()) or ('batch_size' in globals())) and (('num_task' in locals()) or ('num_task' in globals())):
        if num_task > 1:
            indices = [list(range(num_train[x])) for x in range(num_task)]
        else:
            indices = [list(range(num_train[0]))]

    best_valid_accuracy, test_accuracy_at_best_epoch, best_epoch, epoch_bias = 0.0, 0.0, -1, 0
    train_accuracy_hist, valid_accuracy_hist, test_accuracy_hist, best_test_accuracy_hist = [], [], [], []
    task_for_train, task_change_epoch = 0, [1]
    best_param = []

    if not save_param:
        print("Not saving trained parameters")
    else:
        print("Saving trained parameters at '%s'" %(param_folder_path) )

    start_time = timeit.default_timer()
    for train_task_cnt in range(num_task):
        #### Construct new task network on top of earlier task networks
        if train_task_cnt > 0:
            tf.reset_default_graph()
            learning_model.new_lifelong_task(params=param_of_prev_columns)
            del param_of_prev_columns

        model_train_accuracy, model_valid_accuracy, model_test_accuracy = learning_model.train_accuracy, learning_model.valid_accuracy, learning_model.test_accuracy

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            if save_graph:
                tfboard_writer = tf.summary.FileWriter('./graphs/prog_nn/task_' + str(train_task_cnt), sess.graph)

            while learning_step < min(learning_step_max, epoch_bias + patience):
                learning_step = learning_step+1

                #### training process
                if learning_step > 0:
                    shuffle(indices[task_for_train])
                    for batch_cnt in range(num_train[task_for_train]//batch_size):
                        batch_train_x = train_data[task_for_train][0][indices[task_for_train][batch_cnt*batch_size:(batch_cnt+1)*batch_size], :]
                        batch_train_y = train_data[task_for_train][1][indices[task_for_train][batch_cnt*batch_size:(batch_cnt+1)*batch_size]]
                        sess.run(learning_model.update, feed_dict={learning_model.model_input: batch_train_x, learning_model.true_output: batch_train_y, learning_model.epoch: learning_step-1, learning_model.dropout_prob: 0.5})

                #### evaluation process
                train_accuracy_tmp = [0.0 for _ in range(num_task)]
                validation_accuracy_tmp = [0.0 for _ in range(num_task)]
                test_accuracy_tmp = [0.0 for _ in range(num_task)]
                for task_cnt in range(task_for_train+1):
                    for batch_cnt in range(num_train[task_cnt]//batch_size):
                        train_accuracy_tmp[task_cnt] = train_accuracy_tmp[task_cnt] + sess.run(model_train_accuracy[task_cnt], feed_dict={learning_model.model_input: train_data[task_cnt][0][batch_cnt*batch_size:(batch_cnt+1)*batch_size, :], learning_model.true_output: train_data[task_cnt][1][batch_cnt*batch_size:(batch_cnt+1)*batch_size], learning_model.dropout_prob: 1.0})
                    train_accuracy_tmp[task_cnt] = train_accuracy_tmp[task_cnt]/((num_train[task_cnt]//batch_size)*batch_size)

                    for batch_cnt in range(num_valid[task_cnt]//batch_size):
                        validation_accuracy_tmp[task_cnt] = validation_accuracy_tmp[task_cnt] + sess.run(model_valid_accuracy[task_cnt], feed_dict={learning_model.model_input: validation_data[task_cnt][0][batch_cnt*batch_size:(batch_cnt+1)*batch_size, :], learning_model.true_output: validation_data[task_cnt][1][batch_cnt*batch_size:(batch_cnt+1)*batch_size], learning_model.dropout_prob: 1.0})
                    validation_accuracy_tmp[task_cnt] = validation_accuracy_tmp[task_cnt]/((num_valid[task_cnt]//batch_size)*batch_size)

                    for batch_cnt in range(num_test[task_cnt]//batch_size):
                        test_accuracy_tmp[task_cnt] = test_accuracy_tmp[task_cnt] + sess.run(model_test_accuracy[task_cnt], feed_dict={learning_model.model_input: test_data[task_cnt][0][batch_cnt*batch_size:(batch_cnt+1)*batch_size, :], learning_model.true_output: test_data[task_cnt][1][batch_cnt*batch_size:(batch_cnt+1)*batch_size], learning_model.dropout_prob: 1.0})
                    test_accuracy_tmp[task_cnt] = test_accuracy_tmp[task_cnt]/((num_test[task_cnt]//batch_size)*batch_size)

                train_accuracy, valid_accuracy, test_accuracy = sum(train_accuracy_tmp)/num_task, sum(validation_accuracy_tmp)/num_task, sum(test_accuracy_tmp)/num_task
                train_accuracy_to_compare, valid_accuracy_to_compare, test_accuracy_to_compare = train_accuracy_tmp[task_for_train], validation_accuracy_tmp[task_for_train], test_accuracy_tmp[task_for_train]
                print('epoch %d - Train : %f, Validation : %f' % (learning_step, abs(train_accuracy_to_compare), abs(valid_accuracy_to_compare)))

                #### when validation accuracy is better than before
                if valid_accuracy_to_compare > best_valid_accuracy:
                    str_temp = ''
                    if valid_accuracy_to_compare > best_valid_accuracy * improvement_threshold:
                        patience = max(patience, (learning_step-epoch_bias)*patience_multiplier)
                        str_temp = '\t<<'
                    best_valid_accuracy, best_epoch = valid_accuracy_to_compare, learning_step
                    test_accuracy_at_best_epoch = test_accuracy_to_compare
                    print('\t\t\t\t\t\t\tTest : %f%s' % (test_accuracy_at_best_epoch, str_temp))

                train_accuracy_hist.append(train_accuracy_tmp + [abs(train_accuracy)])
                valid_accuracy_hist.append(validation_accuracy_tmp + [abs(valid_accuracy)])
                test_accuracy_hist.append(test_accuracy_tmp + [abs(test_accuracy)])
                best_test_accuracy_hist.append(abs(test_accuracy_at_best_epoch))

                #### switch to new task (only for lifelong learning)
                if doLifelong and learning_step >= epoch_bias+min(patience, learning_step_max//num_task) and len(task_training_order) > 0:
                    print('\n\t>>Change to new task!<<\n')

                    if save_param:
                        para_file_name = param_folder_path + '/model_parameter_t%d.mat'%(task_for_train)
                        curr_param = learning_model.get_params_val(sess)
                        savemat(para_file_name, {'parameter': curr_param})

                    # update epoch_bias, task_for_train, task_change_epoch
                    epoch_bias, task_for_train = learning_step, task_training_order.pop(0)
                    task_change_epoch.append(learning_step+1)

                    # initialize best_valid_accuracy, best_epoch, patience
                    patience = train_hyperpara['patience']
                    best_valid_accuracy, best_epoch = 0.0, -1

                    param_of_prev_columns = learning_model.get_prev_net_param(sess=sess)
                    break

            if save_graph:
                tfboard_writer.close()

    end_time = timeit.default_timer()
    print("End of Training")
    print("Time consumption for training : %.2f" %(end_time-start_time))
    if not doLifelong:
        print("Best validation accuracy : %.4f (at epoch %d)" %(abs(best_valid_accuracy), best_epoch))
        print("Test accuracy at that epoch (%d) : %.4f" %(best_epoch, abs(test_accuracy_at_best_epoch)))

    tmp_valid_accuracy_hist = np.array(valid_accuracy_hist)
    chk_epoch = [(task_change_epoch[x], task_change_epoch[x+1]) for x in range(len(task_change_epoch)-1)] + [(task_change_epoch[-1], learning_step+1)]
    tmp_best_valid_accuracy_list = [np.amax(tmp_valid_accuracy_hist[x[0]:x[1], t]) for x, t in zip(chk_epoch, range(num_task))]

    ## Summary of statistics during training and evaluation
    result_summary = {}
    result_summary['training_time'] = end_time - start_time
    result_summary['num_epoch'] = learning_step
    result_summary['best_epoch'] = best_epoch
    result_summary['history_train_accuracy'] = train_accuracy_hist
    result_summary['history_validation_accuracy'] = valid_accuracy_hist
    result_summary['history_test_accuracy'] = test_accuracy_hist
    result_summary['history_best_test_accuracy'] = best_test_accuracy_hist
    result_summary['best_validation_accuracy'] = sum(tmp_best_valid_accuracy_list) / float(len(tmp_best_valid_accuracy_list))
    result_summary['test_accuracy_at_best_epoch'] = abs(test_accuracy_at_best_epoch)
    result_summary['task_changed_epoch'] = task_change_epoch

    return result_summary, learning_model.num_trainable_var



#### module of training/testing one model
def train_den_net(model_architecture, model_hyperpara, train_hyperpara, dataset, data_type, doLifelong, useGPU=False, GPU_device=0, save_param=False, param_folder_path='saved_param', save_param_interval=100, save_graph=False):
    assert (('den' in model_architecture or 'dynamically' in model_architecture) and doLifelong), "Use train function appropriate to the architecture (Dynamically Expandable Net)"
    print("\nTrain function for Dynamically Expandable Net is called!\n")

    config = tf.ConfigProto()
    if useGPU:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU_device)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.90
        print("GPU %d is used" %(GPU_device))
    else:
        print("CPU is used")

    ## This order of tasks for training can be arbitrary.
    task_training_order = list(range(train_hyperpara['num_tasks']))
    task_for_train, task_change_epoch = task_training_order.pop(0), [1]

    ### set-up data
    train_data, validation_data, test_data = dataset
    if 'mnist' in data_type:
        num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth = mnist_data_print_info(train_data, validation_data, test_data, True, print_info=False)
    elif 'cifar10' in data_type:
        num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth = cifar_data_print_info(train_data, validation_data, test_data, True, print_info=False)
    elif data_type == 'cifar100':
        num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth = cifar_data_print_info(train_data, validation_data, test_data, True, print_info=False)
    elif 'officehome' in data_type:
        num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth = officehome_data_print_info(train_data, validation_data, test_data, True, print_info=False)

    ### reformat data for DEN
    trainX, trainY = [train_data[t][0] for t in task_training_order], [train_data[t][1] for t in task_training_order]
    validX, validY = [validation_data[t][0] for t in task_training_order], [validation_data[t][1] for t in task_training_order]
    testX, testY = [test_data[t][0] for t in task_training_order], [test_data[t][1] for t in task_training_order]

    if save_graph:
        if 'graphs' not in os.listdir(os.getcwd()):
            os.mkdir(os.getcwd()+'/graphs')

    ### Set hyperparameter related to training process
    learning_step_max = train_hyperpara['learning_step_max']
    improvement_threshold = train_hyperpara['improvement_threshold']
    patience = train_hyperpara['patience']
    patience_multiplier = train_hyperpara['patience_multiplier']
    if 'batch_size' in model_hyperpara:
        batch_size = model_hyperpara['batch_size']

    ### Generate Model
    learning_model, generation_success = model_generation(model_architecture, model_hyperpara, train_hyperpara, [x_dim, y_dim, y_depth, num_task])
    if not generation_success:
        return (None, None, None, None)

    learning_model.set_sess_config(config)

    params = dict()
    train_accuracy, valid_accuracy, test_accuracy, best_test_accuracy = [], [], [], []
    start_time = timeit.default_timer()
    for train_task_cnt in range(num_task):
        print("\n\nStart training new task %d" %(train_task_cnt))
        data = (trainX, trainY, validX, validY, testX, testY)

        learning_model.sess = tf.Session(config=config)

        learning_model.T = learning_model.T + 1
        learning_model.task_indices.append(train_task_cnt+1)
        learning_model.load_params(params, time = 1)
        tr_acc, v_acc, te_acc, best_te_acc = learning_model.add_task(train_task_cnt+1, data, save_param, save_graph)
        train_accuracy = train_accuracy+tr_acc
        valid_accuracy = valid_accuracy+v_acc
        test_accuracy = test_accuracy+te_acc
        best_test_accuracy = best_test_accuracy+best_te_acc

        params = learning_model.get_params()

        learning_model.destroy_graph()
        learning_model.sess.close()

    num_trainable_var = learning_model.num_trainable_var(params_list=params)
    end_time = timeit.default_timer()
    print("End of Training")
    print("Time consumption for training : %.2f" %(end_time-start_time))

    task_change_epoch = learning_model.task_change_epoch

    tmp_valid_acc_hist = np.array(valid_accuracy)
    chk_epoch = [(task_change_epoch[x], task_change_epoch[x+1]) for x in range(len(task_change_epoch)-1)] # + [(task_change_epoch[-1], learning_step+1)]
    tmp_best_valid_acc_list = [np.amax(tmp_valid_acc_hist[x[0]:x[1], t]) for x, t in zip(chk_epoch, range(num_task))]

    ## Summary of statistics during training and evaluation
    result_summary = {}
    result_summary['training_time'] = end_time - start_time
    result_summary['num_epoch'] = learning_model.num_training_epoch
    result_summary['history_train_accuracy'] = np.array(train_accuracy)
    result_summary['history_validation_accuracy'] = np.array(valid_accuracy)
    result_summary['history_test_accuracy'] = np.array(test_accuracy)
    result_summary['history_best_test_accuracy'] = np.array(best_test_accuracy)
    result_summary['best_validation_accuracy'] = sum(tmp_best_valid_acc_list) / float(len(tmp_best_valid_acc_list))
    result_summary['test_accuracy_at_best_epoch'] = 0.0
    result_summary['task_changed_epoch'] = task_change_epoch[:-1]
    return result_summary, num_trainable_var
