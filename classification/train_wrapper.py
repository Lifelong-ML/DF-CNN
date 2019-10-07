import sys
from os import listdir, mkdir, remove
from os.path import isfile
from math import sqrt

import scipy.io as spio
import numpy as np
from tensorflow import reset_default_graph

if sys.version_info.major < 3:
    from train import train, train_progressive_net, train_den_net
else:
    from classification.train import train, train_progressive_net, train_den_net

############################################
#### functions to load training results ####
############################################
def loadmat(filename):
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

#### checks if entries in dictionary are mat-objects
def _check_keys(dict):
    for key in dict:
        if isinstance(dict[key], np.ndarray):
            if isinstance(dict[key][0], spio.matlab.mio5_params.mat_struct):
                #### cell of structure case
                tmp = np.zeros((len(dict[key]),), dtype=np.object)
                for cnt in range(len(dict[key])):
                    if sys.version_info.major < 3:
                        tmp[cnt] = _todict2(dict[key][cnt])
                    else:
                        tmp[cnt] = _todict3(dict[key][cnt])
                dict[key] = tmp
            else:
                #### just array case
                dict[key] = list(dict[key])
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            if sys.version_info.major < 3:
                dict[key] = _todict2(dict[key])
            else:
                dict[key] = _todict3(dict[key])
    return dict

#### recursive function constructing nested dictionaries from matobjects
def _todict2(matobj):
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict2(elem)
        elif isinstance(elem, unicode):
            dict[strg] = str(elem)
        elif isinstance(elem, np.ndarray):
            cell_struct = False
            if len(elem) > 0:
                if isinstance(elem[0], spio.matlab.mio5_params.mat_struct):
                    cell_struct = True
            if cell_struct:
                #### cell of structure case
                tmp = np.zeros((len(elem),), dtype=np.object)
                for cnt in range(len(elem)):
                    tmp[cnt] = _todict2(elem[cnt])
                dict[strg] = tmp
            else:
                #### just array case
                dict[strg] = list(elem)
        elif strg == 'hidden_layer':
            dict[strg] = [elem]
        else:
            dict[strg] = elem
    return dict

def _todict3(matobj):
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict3(elem)
        elif isinstance(elem, str):
            dict[strg] = str(elem)
        elif isinstance(elem, np.ndarray):
            cell_struct = False
            if len(elem) > 0:
                if isinstance(elem[0], spio.matlab.mio5_params.mat_struct):
                    cell_struct = True
            if cell_struct:
                #### cell of structure case
                tmp = np.zeros((len(elem),), dtype=np.object)
                for cnt in range(len(elem)):
                    tmp[cnt] = _todict3(elem[cnt])
                dict[strg] = tmp
            else:
                #### just array case
                dict[strg] = list(elem)
        elif strg == 'hidden_layer':
            dict[strg] = [elem]
        else:
            dict[strg] = elem
    return dict

#### compare two model_info_summary
def check_model_equivalency(model_info1, model_info2):
    check_keys1 = [(model_info1_elem in model_info2) for model_info1_elem in model_info1]
    check_keys2 = [(model_info2_elem in model_info1) for model_info2_elem in model_info2]
    if all(check_keys1) and all(check_keys2):
        return all([(model_info1[key] == model_info2[key]) for key in model_info1])
    else:
        return False

def print_model_info(model_dict):
    key_list = list(model_dict.keys())
    key_list.sort()
    for key in key_list:
        print(key + " : ", model_dict[key])

############################################
#### functions to save training results ####
############################################
def mean_of_list(list_input):
    return float(sum(list_input))/len(list_input)

def stddev_of_list(list_input):
    list_mean = mean_of_list(list_input)
    sq_err = [(x-list_mean)**2 for x in list_input]

    if len(list_input)<2:
        return 0.0
    else:
        return sqrt(sum(sq_err)/float(len(list_input)-1))

### Summarize model hyper-parameters in MATLAB structure array
def model_info_summary(model_architecture, model_hyperpara, train_hyperpara, num_para_in_model=-1):
    tmp_dict = {}
    tmp_dict['architecture'] = model_architecture
    tmp_dict['learning_rate'] = train_hyperpara['lr']
    tmp_dict['improvement_threshold'] = train_hyperpara['improvement_threshold']
    tmp_dict['early_stopping_para'] = [train_hyperpara['patience'], train_hyperpara['patience_multiplier']]

    for model_hyperpara_elem in model_hyperpara:
        tmp_dict[model_hyperpara_elem] = model_hyperpara[model_hyperpara_elem]

    if num_para_in_model > -1:
        tmp_dict['number_of_trainable_parameters'] = num_para_in_model
    return tmp_dict

### Compute statistics of multiple independent experiments
def reformat_result_for_mat(model_architecture, model_hyperpara, train_hyperpara, result_from_train_run, data_group_list, num_para_in_model=-1, doLifelong=False):
    result_of_curr_run = {}
    #### 'model_specific_info' element
    result_of_curr_run['model_specific_info'] = model_info_summary(model_architecture, model_hyperpara, train_hyperpara, num_para_in_model)

    num_run_per_model, best_valid_accuracy_list, best_test_accuracy_list, train_time_list = len(result_from_train_run), [], [], []
    result_of_curr_run['result_of_each_run'] = np.zeros((num_run_per_model,), dtype=np.object)
    if doLifelong:
        best_test_accuracy_each_task_list, test_accuracy_at_last_epoch_list = [], []
        best_test_accuracy_each_task_avg_list, test_accuracy_at_last_epoch_avg_list = [], []

    for cnt in range(num_run_per_model):
        result_of_curr_run['result_of_each_run'][cnt] = result_from_train_run[cnt]
        best_valid_accuracy_list.append(result_from_train_run[cnt]['best_validation_accuracy'])
        best_test_accuracy_list.append(result_from_train_run[cnt]['test_accuracy_at_best_epoch'])
        train_time_list.append(result_from_train_run[cnt]['training_time'])
        if doLifelong:
            chk_epoch = [x-1 for x in result_from_train_run[cnt]['task_changed_epoch'][1:]]+[result_from_train_run[cnt]['num_epoch']]
            best_test_accuracy_each_task_list.append([result_from_train_run[cnt]['history_best_test_accuracy'][x] for x in chk_epoch])
            best_test_accuracy_each_task_avg_list.append(mean_of_list(best_test_accuracy_each_task_list[cnt]))

            test_accuracy_at_last_epoch_list.append(result_from_train_run[cnt]['history_test_accuracy'][-1][0:-1])
            test_accuracy_at_last_epoch_avg_list.append(mean_of_list(test_accuracy_at_last_epoch_list[cnt]))

    result_of_curr_run['best_valid_accuracy'] = best_valid_accuracy_list
    result_of_curr_run['best_valid_accuracy_mean'] = mean_of_list(best_valid_accuracy_list)
    result_of_curr_run['best_valid_accuracy_stddev'] = stddev_of_list(best_valid_accuracy_list)
    result_of_curr_run['best_test_accuracy'] = best_test_accuracy_list
    result_of_curr_run['best_test_accuracy_mean'] = mean_of_list(best_test_accuracy_list)
    result_of_curr_run['best_test_accuracy_stddev'] = stddev_of_list(best_test_accuracy_list)
    result_of_curr_run['training_time'] = train_time_list
    result_of_curr_run['training_time_mean'] = mean_of_list(train_time_list)
    result_of_curr_run['training_time_stddev'] = stddev_of_list(train_time_list)
    result_of_curr_run['train_valid_data_group'] = data_group_list
    if doLifelong:
        result_of_curr_run['best_test_acc_each_task'] = best_test_accuracy_each_task_list
        result_of_curr_run['best_test_acc_each_task_mean'] = mean_of_list(best_test_accuracy_each_task_avg_list)
        result_of_curr_run['best_test_acc_each_task_std'] = stddev_of_list(best_test_accuracy_each_task_avg_list)
        result_of_curr_run['test_acc_at_last_epoch'] = test_accuracy_at_last_epoch_list
        result_of_curr_run['test_acc_at_last_epoch_mean'] = mean_of_list(test_accuracy_at_last_epoch_avg_list)
        result_of_curr_run['test_acc_at_last_epoch_std'] = stddev_of_list(test_accuracy_at_last_epoch_avg_list)

    return result_of_curr_run


############################################
#### functions to run several training  ####
####       with same model setting      ####
############################################
def train_run_for_each_model(model_architecture, model_hyperpara, train_hyperpara, dataset, data_type, mat_file_name, saved_result=None, useGPU=False, GPU_device=0, doLifelong=False, saveParam=False, saveParamDir=None, saveGraph=False, tfInitParamPath=None):
    max_run_cnt = train_hyperpara['num_run_per_model']

    #### process results of previous training
    if (saved_result is None) and not (isfile('./Result/'+mat_file_name)):
        saved_result = np.zeros((1,), dtype=np.object)
    elif (saved_result is None):
        saved_result_tmp = spio.loadmat('./Result/'+mat_file_name)
        num_prev_test_model = len(saved_result_tmp['training_summary'][0])
        saved_result = np.zeros((num_prev_test_model+1,), dtype=np.object)
        for cnt in range(num_prev_test_model):
            saved_result[cnt] = saved_result_tmp['training_summary'][0][cnt]
    else:
        num_prev_result, prev_result_tmp = len(saved_result), saved_result
        saved_result = np.zeros((num_prev_result+1,), dtype=np.object)
        for cnt in range(num_prev_result):
            saved_result[cnt] = prev_result_tmp[cnt]

    #### process results saved in temporary file
    if not (isfile('./Result/temp_'+mat_file_name)):
        temp_result_exist = False
    else:
        saved_temp_result_tmp = loadmat('./Result/temp_'+mat_file_name)
        saved_temp_result = saved_temp_result_tmp['training_summary_temp']
        print('\nModel in temp file')
        print_model_info(saved_temp_result['model_specific_info'])
        print('\nModel for current train')
        print_model_info(model_info_summary(model_architecture, model_hyperpara, train_hyperpara))
        use_temp_result = 't'
        if (use_temp_result is 'T') or (use_temp_result is 't') or (use_temp_result is 'True') or (use_temp_result is 'true'):
            temp_result_exist = True
        else:
            temp_result_exist = False

    result_from_train_run = []
    group_cnt = train_hyperpara['train_valid_data_group']
    if temp_result_exist:
        if not (isinstance(saved_temp_result['train_valid_data_group'], np.ndarray) or isinstance(saved_temp_result['train_valid_data_group'], list)):
            result_from_train_run.append(saved_temp_result['result_of_each_run'])
        else:
            for elem in saved_temp_result['result_of_each_run']:
                result_from_train_run.append(elem)
        run_cnt_init = len(result_from_train_run)
        print("\nTemporary Result Exists! Start from %d\n" %(run_cnt_init+1))
    else:
        run_cnt_init = 0
        print("\nTemporary Result is Discarded! Overwrite Temp File\n")

    #### run training procedure with different dataset
    for run_cnt in range(run_cnt_init, max_run_cnt):
        if ('officehome' in data_type):
            train_data, valid_data, test_data = dataset[0][group_cnt[run_cnt]], dataset[1][group_cnt[run_cnt]], dataset[2][group_cnt[run_cnt]]
        else:
            train_data, valid_data, test_data = dataset[0][group_cnt[run_cnt]], dataset[1][group_cnt[run_cnt]], dataset[2]

        save_param_path = None
        if saveParam:
            save_param_path = saveParamDir+'/run'+str(run_cnt)
            mkdir(save_param_path)

        if ('tensorfactor' in model_architecture) and (tfInitParamPath is not None):
            assert ('best_model_parameter.mat' in listdir(tfInitParamPath+'/run'+str(run_cnt))), "(TensorFactor) Parameter file to load does not exist!"
            param_file_to_load = tfInitParamPath+'/run'+str(run_cnt)+'/best_model_parameter.mat'
            tf_init_params = loadmat(param_file_to_load)['parameter']
            print("\tLoad parameter file to initialize TensorFactor model\n")
        else:
            tf_init_params = None

        print("Training/Validation data group : %d\n" %(group_cnt[run_cnt]))
        if 'progressive' in model_architecture:
            train_result_tmp, num_model_para = train_progressive_net(model_architecture, model_hyperpara, train_hyperpara, [train_data, valid_data, test_data], data_type, doLifelong, useGPU, GPU_device, save_param=saveParam, param_folder_path=save_param_path, save_graph=(saveGraph and run_cnt < 1))
        elif ('den' in model_architecture) or ('dynamically' in model_architecture):
            train_result_tmp, num_model_para = train_den_net(model_architecture, model_hyperpara, train_hyperpara, [train_data, valid_data, test_data], data_type, doLifelong, useGPU, GPU_device, save_param=saveParam, param_folder_path=save_param_path, save_graph=(saveGraph and run_cnt < 1))
        else:
            train_result_tmp, num_model_para = train(model_architecture, model_hyperpara, train_hyperpara, [train_data, valid_data, test_data], data_type, doLifelong, useGPU, GPU_device, save_param=saveParam, param_folder_path=save_param_path, save_graph=(saveGraph and run_cnt < 1), tfInitParam=tf_init_params)
        result_from_train_run.append(train_result_tmp)

        if run_cnt < max_run_cnt-1:
            result_of_curr_run = reformat_result_for_mat(model_architecture, model_hyperpara, train_hyperpara, result_from_train_run, group_cnt[0:run_cnt+1], num_model_para, doLifelong)
            spio.savemat('./Result/temp_'+mat_file_name, {'training_summary_temp':result_of_curr_run})
        elif isfile('./Result/temp_'+mat_file_name):
            remove('./Result/temp_'+mat_file_name)

        print("%d-th training run complete\n\n" % (run_cnt+1))
        reset_default_graph()

    #### save training summary
    result_of_curr_run = reformat_result_for_mat(model_architecture, model_hyperpara, train_hyperpara, result_from_train_run, group_cnt[0:max_run_cnt], num_model_para, doLifelong)
    saved_result[-1] = result_of_curr_run
    spio.savemat('./Result/'+mat_file_name, {'training_summary':saved_result})

    return saved_result

