
def num_data_points(data_type, data_percent):
    # the number of images in each sub-task for training, validation, and test
    if 'mnist5' in data_type:
        if data_percent == 1:
            return (100, 20, 1800)
        elif data_percent == 3:
            return (300, 60, 1800)
        elif data_percent == 5:
            return (500, 100, 1800)
        elif data_percent == 7:
            return (700, 140, 1800)
        elif data_percent == 9:
            return (900, 180, 1800)
        elif data_percent == 10:
            return (1000, 200, 1800)
        elif data_percent == 30:
            return (3000, 600, 1800)
        elif data_percent == 50:
            return (5000, 1000, 1800)
        elif data_percent == 70:
            return (7000, 1000, 1800)
        elif data_percent == 90:
            return (9000, 1000, 1800)
        else:
            return (None, None, None)
    elif 'mnist10' in data_type:
        if data_percent == 1:
            return (100, 10, 2000)
        elif data_percent == 3:
            return (300, 30, 2000)
        elif data_percent == 5:
            return (500, 50, 2000)
        elif data_percent == 7:
            return (700, 70, 2000)
        elif data_percent == 9:
            return (900, 90, 2000)
        elif data_percent == 10:
            return (1000, 100, 2000)
        elif data_percent == 30:
            return (3000, 300, 2000)
        elif data_percent == 50:
            return (5000, 500, 2000)
        elif data_percent == 70:
            return (7000, 700, 2000)
        elif data_percent == 90:
            return (9000, 900, 2000)
        else:
            return (None, None, None)
    elif ('cifar10' in data_type) and not ('cifar100' in data_type):
        if data_percent == 2:
            return (160, 40, 2000)
        elif data_percent == 4:
            return (340, 60, 2000)
        elif data_percent == 6:
            return (500, 100, 2000)
        elif data_percent == 8:
            return (660, 140, 2000)
        elif data_percent == 10:
            return (840, 160, 2000)
        elif data_percent == 30:
            return (2500, 500, 2000)
        elif data_percent == 50:
            return (4160, 840, 2000)
        elif data_percent == 70:
            return (5840, 1160, 2000)
        elif data_percent == 90:
            return (7500, 1500, 2000)
        elif data_percent == 100:
            return (8330, 1670, 2000)
        else:
            return (None, None, None)
    elif 'cifar100' in data_type:
        # 10 classes per task
        if data_percent == 2:
            return (80, 20, 1000)
        elif data_percent == 4:
            return (170, 30, 1000)
        elif data_percent == 6:
            return (250, 50, 1000)
        elif data_percent == 8:
            return (330, 70, 1000)
        elif data_percent == 10:
            return (410, 90, 1000)
        elif data_percent == 20:
            return (830, 170, 1000)
        elif data_percent == 25:
            return (1040, 210, 1000)
        elif data_percent == 30:
            return (1250, 250, 1000)
        elif data_percent == 40:
            return (1670, 330, 1000)
        elif data_percent == 50:
            return (2080, 420, 1000)
        elif data_percent == 60:
            return (2500, 500, 1000)
        elif data_percent == 70:
            return (2920, 580, 1000)
        elif data_percent == 75:
            return (3120, 630, 1000)
        elif data_percent == 80:
            return (3330, 670, 1000)
        elif data_percent == 90:
            return (3750, 750, 1000)
        elif data_percent == 100:
            return (4170, 830, 1000)
        else:
            return (None, None, None)


## Hyper-parameters of networks
def model_setup(data_type, data_input_dim, model_type, cnn_padding_type_same=True):
    model_architecture = None
    model_hyperpara = {}
    if cnn_padding_type_same:
        model_hyperpara['padding_type'] = 'SAME'
    else:
        model_hyperpara['padding_type'] = 'VALID'
    model_hyperpara['max_pooling'] = True
    model_hyperpara['dropout'] = True
    model_hyperpara['image_dimension'] = data_input_dim

    if 'mnist' in data_type:
        model_hyperpara['batch_size'] = 10
        if 'ffnn' in model_type.lower():
            model_hyperpara['hidden_layer'] = [256, 128]
        else:
            ## CNN-FFNN case
            model_hyperpara['hidden_layer'] = [32]
        model_hyperpara['kernel_sizes'] = [5, 5, 5, 5]
        model_hyperpara['stride_sizes'] = [1, 1, 1, 1]
        model_hyperpara['channel_sizes'] = [32, 64]
        model_hyperpara['pooling_size'] = [2, 2, 2, 2]
    elif ('cifar10' in data_type) and not ('cifar100' in data_type):
        model_hyperpara['batch_size'] = 20
        if 'ffnn' in model_type.lower():
            model_hyperpara['hidden_layer'] = [256, 64]
        else:
            ## CNN-FFNN case
            model_hyperpara['hidden_layer'] = [64]
        model_hyperpara['kernel_sizes'] = [3, 3, 3, 3, 3, 3, 3, 3]
        model_hyperpara['stride_sizes'] = [1, 1, 1, 1, 1, 1, 1, 1]
        model_hyperpara['channel_sizes'] = [32, 32, 64, 64]
        model_hyperpara['pooling_size'] = [1, 1, 2, 2, 1, 1, 2, 2]
    elif 'cifar100' in data_type:
        model_hyperpara['batch_size'] = 10
        if 'ffnn' in model_type.lower():
            model_hyperpara['hidden_layer'] = [256, 64]
        else:
            ## CNN-FFNN case
            model_hyperpara['hidden_layer'] = [64]
        model_hyperpara['kernel_sizes'] = [3, 3, 3, 3, 3, 3, 3, 3]
        model_hyperpara['stride_sizes'] = [1, 1, 1, 1, 1, 1, 1, 1]
        model_hyperpara['channel_sizes'] = [32, 32, 64, 64]
        model_hyperpara['pooling_size'] = [1, 1, 2, 2, 1, 1, 2, 2]
    elif 'officehome' in data_type:
        model_hyperpara['batch_size'] = 16
        model_hyperpara['hidden_layer'] = [256, 64]
        model_hyperpara['kernel_sizes'] = [11, 11, 5, 5, 3, 3, 3, 3]
        model_hyperpara['stride_sizes'] = [1, 1, 1, 1, 1, 1, 1, 1]
        model_hyperpara['channel_sizes'] = [64, 256, 256, 256]
        model_hyperpara['pooling_size'] = [3, 3, 3, 3, 2, 2, 2, 2]

    if model_type.lower() == 'stl':
        ## Single-task learners
        model_architecture = 'mtl_several_cnn_minibatch'

    elif model_type.lower() == 'snn':
        ## Single neural net for all tasks
        model_architecture = 'mtl_cnn_minibatch'

    elif model_type.lower() == 'hps':
        ## Hard-parameter shared network
        model_architecture = 'mtl_cnn_hps_minibatch'

    elif model_type.lower() == 'tf':
        ## Tensor factorization for multi-task learning
        model_architecture = 'mtl_cnn_tensorfactor_minibatch'
        model_hyperpara['tensor_factor_type'] = 'Tucker'
        model_hyperpara['tensor_factor_error_threshold'] = 1e-3

    elif model_type.lower() == 'prog' or model_type.lower() == 'prognn':
        ## Progressive neural network
        model_architecture = 'll_cnn_progressive_minibatch'
        model_hyperpara['dim_reduction_scale'] = 2.0

    elif model_type.lower() == 'den':
        ## Dynamically expandable network
        model_architecture = 'll_cnn_dynamically_expandable_minibatch'
        model_hyperpara['l1_lambda'] = 1e-6
        model_hyperpara['l2_lambda'] = 0.01
        model_hyperpara['gl_lambda'] = 0.8
        model_hyperpara['reg_lambda'] = 0.5
        model_hyperpara['loss_threshold'] = 0.01
        model_hyperpara['sparsity_threshold'] = [0.98, 0.01]
        model_hyperpara['den_expansion'] = 16

    elif (model_type.lower() == 'dfcnn'):
        ## Deconvolutional factorized CNN in paper 'Learning shared knowledge for deep lifelong learning using deconvolutional networks' of IJCAI 2019
        model_architecture = 'deconvolutional_factorized_cnn'

        if data_type == 'mnist5':
            model_hyperpara['cnn_KB_sizes'] = [3, 12, 3, 24]
            model_hyperpara['cnn_TS_sizes'] = [3, 24, 3, 48]
            model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2]
        elif data_type == 'mnist10':
            model_hyperpara['cnn_KB_sizes'] = [3, 12, 3, 40]
            model_hyperpara['cnn_TS_sizes'] = [3, 24, 3, 64]
            model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2]
        elif ('cifar10' in data_type) and not ('cifar100' in data_type):
            model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2, 2, 2, 2, 2]
            model_hyperpara['cnn_KB_sizes'] = [2, 16, 2, 24, 2, 32, 2, 36]
            model_hyperpara['cnn_TS_sizes'] = [3, 24, 3, 48, 3, 64, 3, 72]
        elif 'cifar100' in data_type:
            model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2, 2, 2, 2, 2]
            model_hyperpara['cnn_KB_sizes'] = [2, 16, 2, 24, 2, 32, 2, 36]
            model_hyperpara['cnn_TS_sizes'] = [3, 24, 3, 48, 3, 64, 3, 72]
        elif 'officehome' in data_type:
            model_hyperpara['cnn_KB_sizes'] = [6, 24, 3, 64, 2, 64, 2, 64]
            model_hyperpara['cnn_TS_sizes'] = [3, 48, 3, 128, 3, 128, 3, 128]
            model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2, 2, 2, 2, 2]

    elif model_type.lower() == 'dfcnn_direct':
        ## Ablated architectures of deconvolutional factorized CNN
        model_architecture = 'deconvolutional_factorized_cnn_direct'

        if data_type == 'mnist5':
            model_hyperpara['cnn_KB_sizes'] = [3, 24, 3, 48]
            model_hyperpara['cnn_TS_sizes'] = [3, 3]
            model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2]
        elif data_type == 'mnist10':
            model_hyperpara['cnn_KB_sizes'] = [3, 24, 3, 64]
            model_hyperpara['cnn_TS_sizes'] = [3, 3]
            model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2]
        elif data_type == 'cifar10_5':
            model_hyperpara['cnn_KB_sizes'] = [2, 24, 2, 48, 2, 64, 2, 72]
            model_hyperpara['cnn_TS_sizes'] = [3, 3, 3, 3]
            model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2, 2, 2, 2, 2]
        elif data_type == 'cifar10_10':
            model_hyperpara['cnn_KB_sizes'] = [2, 24, 2, 48, 2, 64, 2, 72]
            model_hyperpara['cnn_TS_sizes'] = [3, 3, 3, 3]
            model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2, 2, 2, 2, 2]
        elif 'cifar100' in data_type:
            model_hyperpara['cnn_KB_sizes'] = [2, 24, 2, 48, 2, 64, 2, 72]
            model_hyperpara['cnn_TS_sizes'] = [3, 3, 3, 3]
            model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2, 2, 2, 2, 2]

    elif (model_type.lower() == 'dfcnn_tc2'):
        ## Ablated architectures of deconvolutional factorized CNN
        model_architecture = 'deconvolutional_factorized_cnn_tc2'

        if data_type == 'mnist5':
            model_hyperpara['cnn_KB_sizes'] = [3, 1, 12, 3, 16, 24]
            model_hyperpara['cnn_TS_sizes'] = [3, 24, 3, 48]
            model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2]
        elif data_type == 'mnist10':
            model_hyperpara['cnn_KB_sizes'] = [3, 1, 12, 3, 16, 24]
            model_hyperpara['cnn_TS_sizes'] = [3, 24, 3, 48]
            model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2]
        elif data_type == 'cifar10_5':
            model_hyperpara['cnn_KB_sizes'] = [2, 2, 12, 2, 12, 12, 2, 12, 24, 2, 24, 24]
            model_hyperpara['cnn_TS_sizes'] = [3, 24, 3, 24, 3, 48, 3, 48]
            model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2, 2, 2, 2, 2]
        elif data_type == 'cifar10_10':
            model_hyperpara['cnn_KB_sizes'] = [2, 2, 12, 2, 12, 12, 2, 12, 24, 2, 24, 24]
            model_hyperpara['cnn_TS_sizes'] = [3, 24, 3, 24, 3, 48, 3, 48]
            model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2, 2, 2, 2, 2]
        elif 'cifar100' in data_type:
            model_hyperpara['cnn_KB_sizes'] = [2, 2, 12, 2, 12, 12, 2, 12, 24, 2, 24, 24]
            model_hyperpara['cnn_TS_sizes'] = [3, 24, 3, 24, 3, 48, 3, 48]
            model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2, 2, 2, 2, 2]

    else:
        model_hyperpara = None
    return (model_architecture, model_hyperpara)
