from os import getcwd, listdir, mkdir
from utils.utils_env_cl import num_data_points, model_setup
from classification.gen_data import mnist_data, mnist_data_print_info, cifar10_data, cifar100_data, cifar_data_print_info, officehome_data, officehome_data_print_info
from classification.train_wrapper import train_run_for_each_model


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu', help='GPU device ID', type=int, default=-1)
    parser.add_argument('--data_type', help='Type of Data (MNIST5/MNIST10/CIFAR10_5/CIFAR10_10/CIFAR100_10/CIFAR100_20/OfficeHome)', type=str, default='MNIST5')
    parser.add_argument('--data_percent', help='Percentage of train data to be used', type=int, default=50)
    parser.add_argument('--model_type', help='Architecture of Model(STL/SNN/HPS/TF/PROG/DEN/DFCNN/DFCNN_direct/DFCNN_tc2)', type=str, default='STL')
    parser.add_argument('--save_mat_name', help='Name of file to save training results', type=str, default='delete_this.mat')
    parser.add_argument('--cnn_padtype_valid', help='Set CNN padding type VALID', action='store_false', default=True)
    parser.add_argument('--lifelong', help='Train in lifelong learning setting', action='store_true', default=False)
    parser.add_argument('--saveparam', help='Save parameter of NN', action='store_true', default=False)
    parser.add_argument('--savegraph', help='Save graph of NN', action='store_true', default=False)
    parser.add_argument('--tensorfactor_param_path', help='Path to parameters initializing tensor factorized model(below Result, above run0/run1/etc', type=str, default=None)
    args = parser.parse_args()

    gpu_device_num = args.gpu
    if gpu_device_num > -1:
        use_gpu = True
    else:
        use_gpu = False
    do_lifelong = args.lifelong

    if not 'Result' in listdir(getcwd()):
        mkdir('Result')

    ## Name of .mat file recording all information of training and evaluation
    mat_file_name = args.save_mat_name

    data_type, data_percent = args.data_type.lower(), args.data_percent
    data_hyperpara = {}
    data_hyperpara['num_train_group'] = 5   # the number of the set of pre-processed data (each set follows the same experimental setting, but has sets of different images randomly selected.)
    data_hyperpara['multi_class_label'] = False # Binary classification vs multi-class classification

    train_hyperpara = {}
    train_hyperpara['improvement_threshold'] = 1.002      # for accuracy (maximizing it)
    train_hyperpara['patience_multiplier'] = 1.5          # for early-stopping
    if 'mnist' in data_type:
        ## MNIST case
        data_hyperpara['image_dimension'] = [28, 28, 1]
        data_hyperpara['num_train_data'], data_hyperpara['num_valid_data'], data_hyperpara['num_test_data'] = num_data_points(data_type, data_percent)
        if '5' in data_type:
            ## Heterogeneous MTL/LL (each sub-task has distinct set of image classes)
            data_hyperpara['num_tasks'] = 5
        elif '10' in data_type:
            ## Homogeneous MTL/LL (each sub-task has the same set of image classes, but image class set as True only differs)
            data_hyperpara['num_tasks'] = 10
        data_file_name = 'mnist_mtl_data_group_' + str(data_hyperpara['num_train_data']) + '_' + str(data_hyperpara['num_valid_data']) + '_' + str(data_hyperpara['num_test_data']) + '_' + str(data_hyperpara['num_tasks']) + '.pkl'

        train_hyperpara['num_run_per_model'] = 5
        train_hyperpara['train_valid_data_group'] = list(range(5)) + list(range(5))
        train_hyperpara['lr'] = 0.001
        train_hyperpara['lr_decay'] = 1.0/250.0
        train_hyperpara['learning_step_max'] = 500
        train_hyperpara['patience'] = 500

        train_data, validation_data, test_data = mnist_data(data_file_name, data_hyperpara['num_train_data'], data_hyperpara['num_valid_data'], data_hyperpara['num_test_data'], data_hyperpara['num_train_group'], data_hyperpara['num_tasks'], data_percent)
        mnist_data_print_info(train_data, validation_data, test_data)

    elif ('cifar10' in data_type) and not ('cifar100' in data_type):
        ## CIFAR-10 case
        data_hyperpara['image_dimension'] = [32, 32, 3]
        data_hyperpara['num_train_data'], data_hyperpara['num_valid_data'], data_hyperpara['num_test_data'] = num_data_points(data_type, data_percent)
        if '_5' in data_type:
            ## Heterogeneous MTL/LL (each sub-task has distinct set of image classes)
            data_hyperpara['num_tasks'] = 5
        elif '_10' in data_type:
            ## Homogeneous MTL/LL (each sub-task has the same set of image classes, but image class set as True only differs)
            data_hyperpara['num_tasks'] = 10
        data_file_name = 'cifar10_mtl_data_group_' + str(data_hyperpara['num_train_data']) + '_' + str(data_hyperpara['num_valid_data']) + '_' + str(data_hyperpara['num_test_data']) + '_' + str(data_hyperpara['num_tasks']) + '.pkl'

        train_hyperpara['num_run_per_model'] = 5
        train_hyperpara['train_valid_data_group'] = range(5)
        train_hyperpara['lr'] = 0.00025
        train_hyperpara['lr_decay'] = 1.0/1000.0
        train_hyperpara['learning_step_max'] = 2000
        train_hyperpara['patience'] = 2000

        train_data, validation_data, test_data = cifar10_data(data_file_name, data_hyperpara['num_train_data'], data_hyperpara['num_valid_data'], data_hyperpara['num_test_data'], data_hyperpara['num_train_group'], data_hyperpara['num_tasks'], multiclass=data_hyperpara['multi_class_label'], data_percent=data_percent)
        cifar_data_print_info(train_data, validation_data, test_data)

    elif 'cifar100' in data_type:
        ## CIFAR-100 case
        data_hyperpara['multi_class_label'] = True
        data_hyperpara['image_dimension'] = [32, 32, 3]
        data_hyperpara['num_train_data'], data_hyperpara['num_valid_data'], data_hyperpara['num_test_data'] = num_data_points(data_type, data_percent)
        if '_10' in data_type:
            ## Heterogeneous MTL/LL (each sub-task has distinct set of image classes)
            data_hyperpara['num_tasks'] = 10
        elif '_20' in data_type:
            ## Half-homogeneous MTL/LL (there are pairs of sub-tasks which share 5 classes of images)
            data_hyperpara['num_tasks'] = 20
        data_file_name = 'cifar100_mtl_data_group_' + str(data_hyperpara['num_train_data']) + '_' + str(data_hyperpara['num_valid_data']) + '_' + str(data_hyperpara['num_test_data']) + '_' + str(data_hyperpara['num_tasks']) + '.pkl'

        train_hyperpara['num_run_per_model'] = 5
        train_hyperpara['train_valid_data_group'] = range(5)
        train_hyperpara['lr'] = 0.0001
        train_hyperpara['lr_decay'] = 1.0/4000.0
        train_hyperpara['patience'] = 2000
        train_hyperpara['learning_step_max'] = data_hyperpara['num_tasks'] * train_hyperpara['patience']

        train_data, validation_data, test_data = cifar100_data(data_file_name, data_hyperpara['num_train_data'], data_hyperpara['num_valid_data'], data_hyperpara['num_test_data'], data_hyperpara['num_train_group'], data_hyperpara['num_tasks'], multiclass=data_hyperpara['multi_class_label'])
        cifar_data_print_info(train_data, validation_data, test_data)

    elif 'officehome' in data_type:
        ## Office-Home case
        data_hyperpara['multi_class_label'] = True
        data_hyperpara['image_dimension'] = [128, 128, 3]
        data_hyperpara['num_train_data'], data_hyperpara['num_valid_data'], data_hyperpara['num_test_data'] = 0.6, 0.1, 0.3
        data_hyperpara['num_classes'] = 13
        data_hyperpara['num_tasks'] = 10

        data_file_name = 'officehome_mtl_data_group_' + str(data_hyperpara['num_train_data']) + '_' + str(data_hyperpara['num_valid_data']) + '_' + str(data_hyperpara['num_test_data']) + '_t' + str(data_hyperpara['num_tasks']) + '_c' + str(data_hyperpara['num_classes']) + '_i' + str(data_hyperpara['image_dimension'][0]) + '.pkl'

        train_hyperpara['num_run_per_model'] = 5
        train_hyperpara['train_valid_data_group'] = list(range(5)) + list(range(5))
        train_hyperpara['lr'] = 5e-6
        train_hyperpara['lr_decay'] = 1.0/1000.0
        train_hyperpara['patience'] = 1000
        train_hyperpara['learning_step_max'] = data_hyperpara['num_tasks'] * train_hyperpara['patience']

        train_data, validation_data, test_data = officehome_data(data_file_name, data_hyperpara['num_train_data'], data_hyperpara['num_valid_data'], data_hyperpara['num_test_data'], data_hyperpara['num_train_group'], data_hyperpara['image_dimension'])
        officehome_data_print_info(train_data, validation_data, test_data)

    ## Model Set-up
    model_architecture, model_hyperpara = model_setup(data_type, data_hyperpara['image_dimension'], args.model_type, args.cnn_padtype_valid)
    train_hyperpara['num_tasks'] = data_hyperpara['num_tasks']

    save_param_path = None
    if args.saveparam:
        if not 'params' in listdir(getcwd()+'/Result'):
            mkdir('./Result/params')
        save_param_dir_name = data_type + '_' + str(data_percent) + 'p_' + args.model_type
        while save_param_dir_name in listdir(getcwd()+'/Result/params'):
            ## Add dummy characters to directory name to avoid overwriting existing parameter files
            save_param_dir_name += 'a'
        save_param_path = getcwd()+'/Result/params/'+save_param_dir_name
        mkdir(save_param_path)

    print(model_architecture)
    if ('tensorfactor' in model_architecture) and (args.tensorfactor_param_path is not None):
        tensorfactor_param_path = getcwd()+'/Result/'+args.tensorfactor_param_path
    else:
        tensorfactor_param_path = None

    ## Training the Model
    saved_result = train_run_for_each_model(model_architecture, model_hyperpara, train_hyperpara, [train_data, validation_data, test_data], data_type, mat_file_name, saved_result=None, useGPU=use_gpu, GPU_device=gpu_device_num, doLifelong=do_lifelong, saveParam=args.saveparam, saveParamDir=save_param_path, saveGraph=args.savegraph, tfInitParamPath=tensorfactor_param_path)


if __name__ == '__main__':
    main()
