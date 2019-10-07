import sys
import os, pickle
from random import shuffle

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import skimage
import matplotlib.image as mpimg

#### shuffle two lists while maintaining pairs between them
def shuffle_data_x_and_y(data_x, data_y):
    num_x, num_y = data_x.shape[0], data_y.shape[0]
    assert (num_x == num_y), "Given two data have different number of data points"

    indices = list(range(num_x))
    shuffle(indices)
    new_data_x, new_data_y = np.array(data_x[indices]), np.array(data_y[indices])
    return new_data_x, new_data_y


# MNIST data (label : number of train/number of valid/number of test)
#### function to split data into each categories (gather data of same digit)
def mnist_data_class_split(mnist_class):
    train_img, valid_img, test_img = [[] for _ in range(10)], [[] for _ in range(10)], [[] for _ in range(10)]
    for cnt in range(mnist_class.train.images.shape[0]):
        class_inst, x = mnist_class.train.labels[cnt], mnist_class.train.images[cnt, :]
        train_img[class_inst].append(x)

    for cnt in range(mnist_class.validation.images.shape[0]):
        class_inst, x = mnist_class.validation.labels[cnt], mnist_class.validation.images[cnt, :]
        valid_img[class_inst].append(x)

    for cnt in range(mnist_class.test.images.shape[0]):
        class_inst, x = mnist_class.test.labels[cnt], mnist_class.test.images[cnt, :]
        test_img[class_inst].append(x)
    return (train_img, valid_img, test_img)

#### function to shuffle and randomly select some portion of given data
def shuffle_select_some_data(list_of_data, ratio_to_choose):
    #### assume list_of_data = [[class0_img0, class0_img1, ...], [class1_img0, class1_img1, ...], [], ..., []]
    if ratio_to_choose > 1.0:
        ratio_to_choose = float(ratio_to_choose)/100.0

    selected_list_of_data = []
    for class_cnt in range(len(list_of_data)):
        num_data_in_this_class = len(list_of_data[class_cnt])
        num_data_to_choose = int(ratio_to_choose * num_data_in_this_class)

        data_copy = list(list_of_data[class_cnt])
        shuffle(data_copy)
        selected_list_of_data.append(list(data_copy[0:num_data_to_choose]))
    return selected_list_of_data

#### function to concatenate data of image classes which are not specified for exclusion
def concat_data_of_classes(given_data, class_not_to_add):
    data_to_return = []
    for cnt in range(len(given_data)):
        if not (cnt == class_not_to_add):
            data_to_return = data_to_return + given_data[cnt]
    shuffle(data_to_return)
    return data_to_return

#### function to make dataset (either train/valid/test) for binary classification
def mnist_data_gen_binary_classification(img_for_true, img_for_false, dataset_size):
    half_of_dataset_size = dataset_size // 2
    if len(img_for_true) < half_of_dataset_size and len(img_for_false) < half_of_dataset_size:
        return (None, None)
    elif len(img_for_true) < half_of_dataset_size:
        num_true = len(img_for_true)
    elif len(img_for_false) < half_of_dataset_size:
        num_true = dataset_size - len(img_for_false)
    else:
        num_true = half_of_dataset_size

    indices_for_true, indices_for_false = list(range(len(img_for_true))), list(range(len(img_for_false)))
    shuffle(indices_for_true)
    shuffle(indices_for_false)

    indices_classes = [1 for _ in range(num_true)] + [0 for _ in range(dataset_size-num_true)]
    shuffle(indices_classes)

    data_x, data_y, cnt_true = [], [], 0
    for cnt in range(dataset_size):
        if indices_classes[cnt] == 1:
            data_x.append(img_for_true[indices_for_true[cnt_true]])
            data_y.append(1)
            cnt_true = cnt_true+1
        else:
            data_x.append(img_for_false[indices_for_false[cnt - cnt_true]])
            data_y.append(0)
    return (data_x, data_y)

#### function to print information of data file (number of parameters, dimension, etc.)
def mnist_data_print_info(train_data, valid_data, test_data, no_group=False, print_info=True):
    if no_group:
        num_task = len(train_data)

        num_train, num_valid, num_test = [train_data[x][0].shape[0] for x in range(num_task)], [valid_data[x][0].shape[0] for x in range(num_task)], [test_data[x][0].shape[0] for x in range(num_task)]
        x_dim, y_dim = train_data[0][0].shape[1], 0
        y_depth = np.amax(train_data[0][1])+1
        if print_info:
            print("Tasks : %d\nTrain data : %d, Validation data : %d, Test data : %d" %(num_task, num_train, num_valid, num_test))
            print("Input dim : %d, Output dim : %d, Maximum label : %d\n" %(x_dim, y_dim, y_depth))
        return (num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth)
    else:
        assert (len(train_data) == len(valid_data)), "Different number of groups in train/validation data"
        num_group = len(train_data)

        bool_num_task = [(len(train_data[0]) == len(train_data[x])) for x in range(1, num_group)]
        assert all(bool_num_task), "Different number of tasks in some of groups in train data"
        bool_num_task = [(len(valid_data[0]) == len(valid_data[x])) for x in range(1, num_group)]
        assert all(bool_num_task), "Different number of tasks in some of groups in validation data"
        assert (len(train_data[0])==len(valid_data[0]) and len(train_data[0])==len(test_data)), "Different number of tasks in train/validation/test data"
        num_task = len(train_data[0])

        num_train, num_valid, num_test = [train_data[0][x][0].shape[0] for x in range(num_task)], [valid_data[0][x][0].shape[0] for x in range(num_task)], [test_data[x][0].shape[0] for x in range(num_task)]
        x_dim, y_dim = train_data[0][0][0].shape[1], 0
        y_depth = max([np.amax(train_data[0][x][1]) for x in range(num_task)])+1
        if print_info:
            print("Tasks : %d, Groups of training/valid : %d\n" %(num_task, num_group))
            print("Train data : ", num_train, ", Validation data : ", num_valid, ", Test data : ", num_test)
            print("Input dim : %d, Output dim : %d, Maximum label : %d\n" %(x_dim, y_dim, y_depth))
        return (num_task, num_group, num_train, num_valid, num_test, x_dim, y_dim, y_depth)


#### generate/handle data of mnist
#### data format (train_data, validation_data, test_data)
####    - train/validation : [group1(list), group2(list), ... ] with the group of test data format
####    - test : [(task1_x, task1_y), (task2_x, task2_y), ... ]
def mnist_data(data_file_name, num_train_max, num_valid_max, num_test_max, num_train_group, num_tasks=5, data_percent=100):
    curr_path = os.getcwd()
    if not ('Data' in os.listdir(curr_path)):
        os.mkdir('./Data')

    data_path = curr_path + '/Data'
    if data_file_name in os.listdir(data_path):
        with open('./Data/' + data_file_name, 'rb') as fobj:
            if sys.version_info.major < 3:
                train_data, validation_data, test_data = pickle.load(fobj)
            else:
                train_data, validation_data, test_data = pickle.load(fobj, encoding='latin1')
            print('Successfully load data')
    else:
        mnist = input_data.read_data_sets('Data/MNIST_data', one_hot=False)
        #### subclasses : train, validation, test with images/labels subclasses
        categorized_train_x, categorized_valid_x, categorized_test_x = mnist_data_class_split(mnist)

        if num_tasks == 5:
            #### split data into completely different multi-task datasets (5 tasks of 0 vs 1, 2 vs 3, ..., 8 vs 9)
            ## process train data
            train_data = []
            for group_cnt in range(num_train_group):
                train_data_tmp = []
                for task_cnt in range(num_tasks):
                    train_x_tmp, train_y_tmp = mnist_data_gen_binary_classification(categorized_train_x[2*task_cnt], categorized_train_x[2*task_cnt+1], num_train_max)
                    train_data_tmp.append( ( np.array(train_x_tmp), np.array(train_y_tmp) ) )
                train_data.append(train_data_tmp)

            ## process validation data
            validation_data = []
            for group_cnt in range(num_train_group):
                validation_data_tmp = []
                for task_cnt in range(num_tasks):
                    valid_x_tmp, valid_y_tmp = mnist_data_gen_binary_classification(categorized_valid_x[2*task_cnt], categorized_valid_x[2*task_cnt+1], num_valid_max)
                    validation_data_tmp.append( ( np.array(valid_x_tmp), np.array(valid_y_tmp) ) )
                validation_data.append(validation_data_tmp)

            ## process test data
            test_data = []
            for task_cnt in range(num_tasks):
                test_x_tmp, test_y_tmp = mnist_data_gen_binary_classification(categorized_test_x[2*task_cnt], categorized_test_x[2*task_cnt+1], num_test_max)
                test_data.append( ( np.array(test_x_tmp), np.array(test_y_tmp) ) )
        elif num_tasks == 10:
            #### split data into 10 tasks of 1-vs-all
            ## process train data
            train_data = []
            for group_cnt in range(num_train_group):
                train_data_tmp = []
                selected_categorized_train_x = shuffle_select_some_data(categorized_train_x, data_percent)

                for task_cnt in range(num_tasks):
                    train_x_tmp, train_y_tmp = mnist_data_gen_binary_classification(selected_categorized_train_x[task_cnt], concat_data_of_classes(selected_categorized_train_x, task_cnt), num_train_max)
                    train_data_tmp.append( ( np.array(train_x_tmp), np.array(train_y_tmp) ) )
                train_data.append(train_data_tmp)

            ## process validation data
            validation_data = []
            for group_cnt in range(num_train_group):
                validation_data_tmp = []
                selected_categorized_valid_x = shuffle_select_some_data(categorized_valid_x, data_percent)
                for task_cnt in range(num_tasks):
                    valid_x_tmp, valid_y_tmp = mnist_data_gen_binary_classification(selected_categorized_valid_x[task_cnt], concat_data_of_classes(selected_categorized_valid_x, task_cnt), num_valid_max)
                    validation_data_tmp.append( ( np.array(valid_x_tmp), np.array(valid_y_tmp) ) )
                validation_data.append(validation_data_tmp)

            ## process test data
            test_data = []
            for task_cnt in range(num_tasks):
                test_x_tmp, test_y_tmp = mnist_data_gen_binary_classification(categorized_test_x[task_cnt], concat_data_of_classes(categorized_test_x, task_cnt), num_test_max)
                test_data.append( ( np.array(test_x_tmp), np.array(test_y_tmp) ) )
        else:
            print('Check number of tasks - MNIST Data gen.')
            return None

        #### save data
        with open('./Data/' + data_file_name, 'wb') as fobj:
            pickle.dump([train_data, validation_data, test_data], fobj)
            print('Successfully generate/save data')
    return (train_data, validation_data, test_data)



############################################################
#### CIFAR dataset
############################################################
#### read raw data from cifar10 files
def read_cifar10_data(data_path):
    train_x_list, train_y_list = [], []
    ## Read training set
    for cnt in range(1, 6):
        file_name = 'data_batch_'+str(cnt)
        with open(data_path+'/'+file_name, 'rb') as fobj:
            if sys.version_info.major < 3:
                dict_tmp = pickle.load(fobj)
            else:
                dict_tmp = pickle.load(fobj, encoding='latin1')
            train_x_tmp, train_y_tmp = dict_tmp['data'], dict_tmp['labels']
            num_data_in_file = train_x_tmp.shape[0]
            train_x = train_x_tmp.reshape(num_data_in_file, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255.0
            train_x = train_x.reshape(num_data_in_file, 3*32*32)
            train_y = np.array(train_y_tmp)

            train_x_list.append(train_x)
            train_y_list.append(train_y)
            print("\tRead %s file" %(file_name))
    proc_train_x, proc_train_y = np.concatenate(train_x_list), np.concatenate(train_y_list)

    ## Read test set
    with open(data_path+'/test_batch', 'rb') as fobj:
        if sys.version_info.major < 3:
            dict_tmp = pickle.load(fobj)
        else:
            dict_tmp = pickle.load(fobj, encoding='latin1')
        test_x_tmp, test_y_tmp = dict_tmp['data'], dict_tmp['labels']
        num_data_in_file = test_x_tmp.shape[0]
        test_x = test_x_tmp.reshape(num_data_in_file, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255.0
        proc_test_x = test_x.reshape(num_data_in_file, 3*32*32).astype(np.float32)
        proc_test_y = np.array(test_y_tmp)
        print("\tRead test_batch file")
    return (proc_train_x, proc_train_y, proc_test_x, proc_test_y)


#### function to split data into each categories (gather data of same digit)
def cifar_data_class_split(cifar_data, num_class):
    raw_train_x, raw_train_y, raw_test_x, raw_test_y = cifar_data
    train_img, test_img = [[] for _ in range(num_class)], [[] for _ in range(num_class)]
    for cnt in range(len(raw_train_x)):
        train_img[int(raw_train_y[cnt])].append(raw_train_x[cnt])

    for cnt in range(len(raw_test_x)):
        test_img[int(raw_test_y[cnt])].append(raw_test_x[cnt])
    return (train_img, test_img)


#### function to split train data into train and validation data
def cifar_split_for_validation_data(categorized_cifar_train_data, train_valid_ratio):
    num_class = len(categorized_cifar_train_data)
    train_img, valid_img = [[] for _ in range(num_class)], [[] for _ in range(num_class)]
    for class_cnt in range(num_class):
        num_data = len(categorized_cifar_train_data[class_cnt])
        indices, num_valid_data = list(range(num_data)), int(num_data*train_valid_ratio/(1.0+train_valid_ratio))
        shuffle(indices)
        for data_cnt in range(num_data):
            if data_cnt < num_valid_data:
                valid_img[class_cnt].append(categorized_cifar_train_data[class_cnt][indices[data_cnt]])
            else:
                train_img[class_cnt].append(categorized_cifar_train_data[class_cnt][indices[data_cnt]])
    return (train_img, valid_img)


#### function to normalize cifar data
def cifar_data_standardization(raw_data):
    if len(raw_data.shape)<2:
        ## single numpy array
        num_elem = raw_data.shape[0]
        raw_data_reshaped = raw_data.reshape(num_elem//3, 3)
        mean, std = np.mean(raw_data_reshaped, dtype=np.float32, axis=0), np.std(raw_data_reshaped, dtype=np.float32, axis=0)
        adjusted_std = np.maximum(std, [1.0/np.sqrt(raw_data_reshaped.shape[0]) for _ in range(3)])
        new_data = ((raw_data_reshaped - mean)/adjusted_std).reshape(num_elem)
    else:
        ## 2D numpy array
        num_data, num_feature = raw_data.shape
        raw_data_reshaped = raw_data.reshape(num_data*num_feature//3, 3)
        mean, std = np.mean(raw_data_reshaped, dtype=np.float32, axis=0), np.std(raw_data_reshaped, dtype=np.float32, axis=0)
        adjusted_std = np.maximum(std, [1.0/np.sqrt(raw_data_reshaped.shape[0]) for _ in range(3)])
        new_data = ((raw_data_reshaped - mean)/adjusted_std).reshape(num_data, num_feature)
    return new_data


#### function to make dataset (either train/valid/test) for binary classification
def cifar_data_gen_binary_classification(img_for_true, img_for_false, dataset_size):
    if dataset_size < 1:
        dataset_size = len(img_for_true) + len(img_for_false)

    num_true = min(dataset_size//2, len(img_for_true))

    indices_for_true, indices_for_false = list(range(len(img_for_true))), list(range(len(img_for_false)))
    shuffle(indices_for_true)
    shuffle(indices_for_false)

    indices_classes = [1 for _ in range(num_true)] + [0 for _ in range(dataset_size-num_true)]
    shuffle(indices_classes)

    data_x, data_y, cnt_false = [], [], 0
    for cnt in range(dataset_size):
        if indices_classes[cnt] == 0:
            img_tmp = img_for_false[indices_for_false[cnt_false]]
            data_x.append(img_tmp)
            data_y.append(0)
            cnt_false = cnt_false+1
        else:
            img_tmp = img_for_true[indices_for_true[cnt-cnt_false]]
            data_x.append(img_tmp)
            data_y.append(1)
    data_x = cifar_data_standardization(np.array(data_x))
    return (data_x, data_y)


#### function to make dataset (either train/valid/test) for multi-class classification
def cifar_data_gen_multiclass_classification(imgs, num_class, dataset_size):
    if dataset_size < 1:
        dataset_size = sum([len(x) for x in imgs])

    num_at_each_class = [int(dataset_size/num_class) for _ in range(num_class)]

    indices_classes, indices_imgs = [], [list(range(len(x))) for x in imgs]
    for cnt in range(num_class):
        indices_classes = indices_classes + [cnt for _ in range(num_at_each_class[cnt])]
        shuffle(indices_imgs[cnt])
    shuffle(indices_classes)

    data_x, data_y, cnt_imgs = [], [], [0 for _ in range(num_class-1)]
    for cnt in range(dataset_size):
        img_label = indices_classes[cnt]
        if img_label > num_class-2:
            img_tmp = imgs[img_label][cnt-sum(cnt_imgs)]
        else:        
            img_tmp = imgs[img_label][cnt_imgs[img_label]]
            cnt_imgs[img_label] = cnt_imgs[img_label]+1
        data_x.append(img_tmp)
        data_y.append(img_label)
    data_x = cifar_data_standardization(np.array(data_x))
    return (data_x, data_y)


#### function to print information of data file (number of parameters, dimension, etc.)
def cifar_data_print_info(train_data, valid_data, test_data, no_group=False, print_info=True):
    if no_group:
        num_task = len(train_data)

        num_train, num_valid, num_test = [train_data[x][0].shape[0] for x in range(num_task)], [valid_data[x][0].shape[0] for x in range(num_task)], [test_data[x][0].shape[0] for x in range(num_task)]
        x_dim, y_dim = train_data[0][0].shape[1], 0
        y_depth = np.amax(train_data[0][1])+1
        if print_info:
            print("Tasks : %d\nTrain data : %d, Validation data : %d, Test data : %d" %(num_task, num_train, num_valid, num_test))
            print("Input dim : %d, Output dim : %d, Maximum label : %d\n" %(x_dim, y_dim, y_depth))
        return (num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth)
    else:
        assert (len(train_data) == len(valid_data)), "Different number of groups in train/validation data"
        num_group = len(train_data)

        bool_num_task = [(len(train_data[0]) == len(train_data[x])) for x in range(1, num_group)]
        assert all(bool_num_task), "Different number of tasks in some of groups in train data"
        bool_num_task = [(len(valid_data[0]) == len(valid_data[x])) for x in range(1, num_group)]
        assert all(bool_num_task), "Different number of tasks in some of groups in validation data"
        assert (len(train_data[0])==len(valid_data[0]) and len(train_data[0])==len(test_data)), "Different number of tasks in train/validation/test data"
        num_task = len(train_data[0])

        num_train, num_valid, num_test = [train_data[0][x][0].shape[0] for x in range(num_task)], [valid_data[0][x][0].shape[0] for x in range(num_task)], [test_data[x][0].shape[0] for x in range(num_task)]
        x_dim, y_dim = train_data[0][0][0].shape[1], 0
        y_depth = max([np.amax(train_data[0][x][1]) for x in range(num_task)])+1
        if print_info:
            print("Tasks : %d, Groups of training/valid : %d\n" %(num_task, num_group))
            print("Train data : ", num_train, ", Validation data : ", num_valid, ", Test data : ", num_test)
            print("Input dim : %d, Output dim : %d, Maximum label : %d\n" %(x_dim, y_dim, y_depth))
        return (num_task, num_group, num_train, num_valid, num_test, x_dim, y_dim, y_depth)


#### process cifar10 data to generate pickle file with data in right format for DNN models
def cifar10_data(data_file_name, num_train_max, num_valid_max, num_test_max, num_train_group, num_tasks, train_valid_ratio=0.2, multiclass=False, data_percent=100):
    curr_path = os.getcwd()
    if not ('Data' in os.listdir(curr_path)):
        os.mkdir('./Data')

    data_path = curr_path + '/Data'
    if data_file_name in os.listdir(data_path):
        with open('./Data/' + data_file_name, 'rb') as fobj:
            if sys.version_info.major < 3:
                train_data, validation_data, test_data = pickle.load(fobj)
            else:
                train_data, validation_data, test_data = pickle.load(fobj, encoding='latin1')
            print('Successfully load data')
    else:
        raw_data_path = curr_path + '/Data/cifar-10-batches-py'
        cifar10_trainx, cifar10_trainy, cifar10_testx, cifar10_testy = read_cifar10_data(raw_data_path)

        #### split data into sets for each label
        categorized_train_x_tmp, categorized_test_x = cifar_data_class_split([cifar10_trainx, cifar10_trainy, cifar10_testx, cifar10_testy], 10)
        categorized_train_x, categorized_valid_x = cifar_split_for_validation_data(categorized_train_x_tmp, train_valid_ratio)

        if multiclass:
            #### make data into multi-class
            ## process train data
            train_data = []
            for group_cnt in range(num_train_group):
                train_x_tmp, train_y_tmp = cifar_data_gen_multiclass_classification(categorized_train_x, 10, num_train_max)
                train_data.append( [( np.array(train_x_tmp), np.array(train_y_tmp) )] )

            ## process validation data
            validation_data = []
            for group_cnt in range(num_train_group):
                valid_x_tmp, valid_y_tmp = cifar_data_gen_multiclass_classification(categorized_valid_x, 10, num_valid_max)
                validation_data.append( [( np.array(valid_x_tmp), np.array(valid_y_tmp) )] )

            ## process test data
            test_data = []
            for group_cnt in range(num_train_group):
                test_x_tmp, test_y_tmp = cifar_data_gen_multiclass_classification(categorized_test_x, 10, num_test_max)
                test_data.append( ( np.array(test_x_tmp), np.array(test_y_tmp) ) )
        else:
            if num_tasks == 5:
                #### split data into completely different multi-task datasets (5 tasks of 0 vs 1, 2 vs 3, ..., 8 vs 9)
                ## process train data
                train_data = []
                for group_cnt in range(num_train_group):
                    train_data_tmp = []
                    for task_cnt in range(num_tasks):
                        train_x_tmp, train_y_tmp = cifar_data_gen_binary_classification(categorized_train_x[2*task_cnt], categorized_train_x[2*task_cnt+1], num_train_max)
                        train_data_tmp.append( ( np.array(train_x_tmp), np.array(train_y_tmp) ) )
                    train_data.append(train_data_tmp)

                ## process validation data
                validation_data = []
                for group_cnt in range(num_train_group):
                    validation_data_tmp = []
                    for task_cnt in range(num_tasks):
                        valid_x_tmp, valid_y_tmp = cifar_data_gen_binary_classification(categorized_valid_x[2*task_cnt], categorized_valid_x[2*task_cnt+1], num_valid_max)
                        validation_data_tmp.append( ( np.array(valid_x_tmp), np.array(valid_y_tmp) ) )
                    validation_data.append(validation_data_tmp)

                ## process test data
                test_data = []
                for task_cnt in range(num_tasks):
                    test_x_tmp, test_y_tmp = cifar_data_gen_binary_classification(categorized_test_x[2*task_cnt], categorized_test_x[2*task_cnt+1], num_test_max)
                    test_data.append( ( np.array(test_x_tmp), np.array(test_y_tmp) ) )

            elif num_tasks == 10:
                #### split data into 10 tasks of one-vs-all multi-task datasets
                ## process train data
                train_data = []
                for group_cnt in range(num_train_group):
                    train_data_tmp = []
                    selected_categorized_train_x = shuffle_select_some_data(categorized_train_x, data_percent)

                    for task_cnt in range(num_tasks):
                        train_x_tmp, train_y_tmp = cifar_data_gen_binary_classification(selected_categorized_train_x[task_cnt], concat_data_of_classes(selected_categorized_train_x, task_cnt), num_train_max)
                        train_data_tmp.append( ( np.array(train_x_tmp), np.array(train_y_tmp) ) )
                    train_data.append(train_data_tmp)

                ## process validation data
                validation_data = []
                for group_cnt in range(num_train_group):
                    validation_data_tmp = []
                    selected_categorized_valid_x = shuffle_select_some_data(categorized_valid_x, data_percent)

                    for task_cnt in range(num_tasks):
                        valid_x_tmp, valid_y_tmp = cifar_data_gen_binary_classification(selected_categorized_valid_x[task_cnt], concat_data_of_classes(selected_categorized_valid_x, task_cnt), num_valid_max)
                        validation_data_tmp.append( ( np.array(valid_x_tmp), np.array(valid_y_tmp) ) )
                    validation_data.append(validation_data_tmp)

                ## process test data
                test_data = []
                for task_cnt in range(num_tasks):
                    test_x_tmp, test_y_tmp = cifar_data_gen_binary_classification(categorized_test_x[task_cnt], concat_data_of_classes(categorized_test_x, task_cnt), num_test_max)
                    test_data.append( ( np.array(test_x_tmp), np.array(test_y_tmp) ) )

        #### save data
        with open('./Data/' + data_file_name, 'wb') as fobj:
            pickle.dump([train_data, validation_data, test_data], fobj)
            print('Successfully generate/save data')
    return (train_data, validation_data, test_data)



################################# CIFAR-100 Data
# [first five tasks]
# 'aquatic_mammals'			4, 30, 55, 72, 95
# 'fish'					1, 32, 67, 73, 91
# 'flowers'				54, 62, 70, 82, 92
# 'food_containers'			9, 10, 16, 28, 61
# 'fruit_and_vegetables'			0, 51, 53, 57, 83
# 'household_furniture'			5, 20, 25, 84, 94
# 'insects'				6, 7, 14, 18, 24
# 'large_carnivores'			3, 42, 43, 88, 97
# 'large_man-made_outdoor_things'		12, 17, 37, 68, 76
# 'vehicles_1'				8, 13, 48, 58, 90
# 
# [second five tasks]
# 'household_electrical_devices'		22, 39, 40, 86, 87
# 'large_natural_outdoor_scenes'		23, 33, 49, 60, 71
# 'large_omnivores_and_herbivores'	15, 19, 21, 31, 38
# 'medium_mammals'			34, 63, 64, 66, 75
# 'non-insect_invertebrates'		26, 45, 77, 79, 99
# 'people'				2, 11, 35, 46, 98
# 'reptiles'				27, 29, 44, 78, 93
# 'small_mammals'				36, 50, 65, 74, 80
# 'trees'					47, 52, 56, 59, 96
# 'vehicles_2'				41, 69, 81, 85, 89

_cifar100_task_labels_10 = [[4, 1, 54, 9, 0, 5, 6, 3, 12, 8],
                            [30, 32, 62, 10, 51, 20, 7, 42, 17, 13],
                            [55, 67, 70, 16, 53, 25, 14, 43, 37, 48],
                            [72, 73, 82, 28, 57, 84, 18, 88, 68, 58],
                            [95, 91, 92, 61, 83, 94, 24, 97, 76, 90],
                            [22, 23, 15, 34, 26, 2, 27, 36, 47, 41],
                            [39, 33, 19, 63, 45, 11, 29, 50, 52, 69],
                            [40, 49, 21, 64, 77, 35, 44, 65, 56, 81],
                            [86, 60, 31, 66, 79, 46, 78, 74, 59, 85],
                            [87, 71, 38, 75, 99, 98, 93, 80, 96, 89]]

_cifar100_task_labels_20 = [[4, 1, 54, 9, 0, 5, 6, 3, 12, 8],
                            [4, 23, 54, 34, 0, 2, 6, 36, 12, 41],
                            [22, 23, 15, 34, 26, 2, 27, 36, 47, 41],
                            [22, 32, 15, 10, 26, 20, 27, 42, 47, 13],
                            [30, 32, 62, 10, 51, 20, 7, 42, 17, 13],
                            [30, 33, 62, 63, 51, 11, 7, 50, 17, 69],
                            [39, 33, 19, 63, 45, 11, 29, 50, 52, 69],
                            [39, 67, 19, 16, 45, 25, 29, 43, 52, 48],
                            [55, 67, 70, 16, 53, 25, 14, 43, 37, 48],
                            [55, 49, 70, 64, 53, 35, 14, 65, 37, 81],
                            [40, 49, 21, 64, 77, 35, 44, 65, 56, 81],
                            [40, 73, 21, 28, 77, 84, 44, 88, 56, 58],
                            [72, 73, 82, 28, 57, 84, 18, 88, 68, 58],
                            [72, 60, 82, 66, 57, 46, 18, 74, 68, 85],
                            [86, 60, 31, 66, 79, 46, 78, 74, 59, 85],
                            [86, 71, 31, 75, 79, 98, 78, 80, 59, 89],
                            [87, 71, 38, 75, 99, 98, 93, 80, 96, 89],
                            [87, 91, 38, 61, 99, 94, 93, 97, 96, 90],
                            [95, 91, 92, 61, 83, 94, 24, 97, 76, 90],
                            [95, 1, 92, 9, 83, 5, 24, 3, 76, 8]]

#### read raw data from cifar100 files
def read_cifar100_data(data_path):
    ## Read training set
    with open(data_path+'/train', 'rb') as fobj:
        if sys.version_info.major < 3:
            dict_tmp = pickle.load(fobj)
        else:
            dict_tmp = pickle.load(fobj, encoding='latin1')
        train_x_tmp, train_y_tmp = dict_tmp['data'], dict_tmp['fine_labels']
        num_data_in_file = train_x_tmp.shape[0]
        train_x = train_x_tmp.reshape(num_data_in_file, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255.0
        proc_train_x = train_x.reshape(num_data_in_file, 3*32*32).astype(np.float32)
        proc_train_y = np.array(train_y_tmp)
        print("\tRead train_batch file")

    ## Read test set
    with open(data_path+'/test', 'rb') as fobj:
        if sys.version_info.major < 3:
            dict_tmp = pickle.load(fobj)
        else:
            dict_tmp = pickle.load(fobj, encoding='latin1')
        test_x_tmp, test_y_tmp = dict_tmp['data'], dict_tmp['fine_labels']
        num_data_in_file = test_x_tmp.shape[0]
        test_x = test_x_tmp.reshape(num_data_in_file, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255.0
        proc_test_x = test_x.reshape(num_data_in_file, 3*32*32).astype(np.float32)
        proc_test_y = np.array(test_y_tmp)
        print("\tRead test_batch file")
    return (proc_train_x, proc_train_y, proc_test_x, proc_test_y)


#### process cifar100 data to generate pickle file with data in right format for DNN models
#### train/test data : 500/100 per class
def cifar100_data(data_file_name, num_train_max, num_valid_max, num_test_max, num_train_group, num_tasks, train_valid_ratio=0.2, multiclass=False):
    curr_path = os.getcwd()
    if not ('Data' in os.listdir(curr_path)):
        os.mkdir('./Data')

    data_path = curr_path + '/Data'
    if data_file_name in os.listdir(data_path):
        with open('./Data/' + data_file_name, 'rb') as fobj:
            if sys.version_info.major < 3:
                train_data, validation_data, test_data = pickle.load(fobj)
            else:
                train_data, validation_data, test_data = pickle.load(fobj, encoding='latin1')
            print('Successfully load data')
    else:
        raw_data_path = curr_path + '/Data/cifar-100-python'
        cifar100_trainx, cifar100_trainy, cifar100_testx, cifar100_testy = read_cifar100_data(raw_data_path)

        #### split data into sets for each label
        categorized_train_x_tmp, categorized_test_x = cifar_data_class_split([cifar100_trainx, cifar100_trainy, cifar100_testx, cifar100_testy], 100)
        categorized_train_x, categorized_valid_x = cifar_split_for_validation_data(categorized_train_x_tmp, train_valid_ratio)

        if multiclass:
            #### make data into multi-class
            ## process train data
            assert (num_tasks == len(_cifar100_task_labels_10) or num_tasks == len(_cifar100_task_labels_20)), "Number of tasks doesn't match the group of class labels"

            if num_tasks == len(_cifar100_task_labels_10):
                _cifar100_task_labels = _cifar100_task_labels_10
            elif num_tasks == len(_cifar100_task_labels_20):
                _cifar100_task_labels = _cifar100_task_labels_20

            train_data = []
            for group_cnt in range(num_train_group):
                train_data_tmp = []
                for task_cnt in range(num_tasks):
                    train_x_tmp, train_y_tmp = cifar_data_gen_multiclass_classification([categorized_train_x[x] for x in _cifar100_task_labels[task_cnt]], len(_cifar100_task_labels[task_cnt]), num_train_max)
                    train_data_tmp.append( ( np.array(train_x_tmp), np.array(train_y_tmp) ) )
                train_data.append(train_data_tmp)

            ## process validation data
            validation_data = []
            for group_cnt in range(num_train_group):
                validation_data_tmp = []
                for task_cnt in range(num_tasks):
                    valid_x_tmp, valid_y_tmp = cifar_data_gen_multiclass_classification([categorized_valid_x[x] for x in _cifar100_task_labels[task_cnt]], len(_cifar100_task_labels[task_cnt]), num_valid_max)
                    validation_data_tmp.append( ( np.array(valid_x_tmp), np.array(valid_y_tmp) ) )
                validation_data.append(validation_data_tmp)

            ## process test data
            test_data = []
            for task_cnt in range(num_tasks):
                test_x_tmp, test_y_tmp = cifar_data_gen_multiclass_classification([categorized_test_x[x] for x in _cifar100_task_labels[task_cnt]], len(_cifar100_task_labels[task_cnt]), num_test_max)
                test_data.append( ( np.array(test_x_tmp), np.array(test_y_tmp) ) )
        else:
            #### split data into completely different multi-task datasets (5 tasks of 0 vs 1, 2 vs 3, ..., 8 vs 9)
            ## process train data
            train_data = []
            for group_cnt in range(num_train_group):
                train_data_tmp = []
                for task_cnt in range(num_tasks):
                    train_x_tmp, train_y_tmp = cifar_data_gen_binary_classification(categorized_train_x[2*task_cnt], categorized_train_x[2*task_cnt+1], num_train_max)
                    train_data_tmp.append( ( np.array(train_x_tmp), np.array(train_y_tmp) ) )
                train_data.append(train_data_tmp)

            ## process validation data
            validation_data = []
            for group_cnt in range(num_train_group):
                validation_data_tmp = []
                for task_cnt in range(num_tasks):
                    valid_x_tmp, valid_y_tmp = cifar_data_gen_binary_classification(categorized_valid_x[2*task_cnt], categorized_valid_x[2*task_cnt+1], num_valid_max)
                    validation_data_tmp.append( ( np.array(valid_x_tmp), np.array(valid_y_tmp) ) )
                validation_data.append(validation_data_tmp)

            ## process test data
            test_data = []
            for task_cnt in range(num_tasks):
                test_x_tmp, test_y_tmp = cifar_data_gen_binary_classification(categorized_test_x[2*task_cnt], categorized_test_x[2*task_cnt+1], num_test_max)
                test_data.append( ( np.array(test_x_tmp), np.array(test_y_tmp) ) )

        #### save data
        with open('./Data/' + data_file_name, 'wb') as fobj:
            pickle.dump([train_data, validation_data, test_data], fobj)
            print('Successfully generate/save data')
    return (train_data, validation_data, test_data)



################################### Office-Home Data
# 65 classes
_officehome_class_labels = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator', 'Calendar',
                            'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp', 'Drill', 'Eraser',
                            'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork', 'Glasses', 'Hammer',
                            'Helmet', 'Kettle', 'Keyboard', 'Knives', 'Lamp_Shade', 'Laptop', 'Marker', 'Monitor', 'Mop',
                            'Mouse', 'Mug', 'Notebook', 'Oven', 'Pan', 'Paper_Clip', 'Pen', 'Pencil', 'Postit_Notes', 'Printer',
                            'Push_Pin', 'Radio', 'Refrigerator', 'Ruler', 'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers',
                            'Soda', 'Speaker', 'Spoon', 'Table', 'Telephone', 'ToothBrush', 'Toys', 'Trash_Can', 'TV', 'Webcam']

# randomly split subtasks
_officehome_task_labels  = [ [63, 54, 55, 59, 47, 8, 5, 56, 18, 50, 3, 57, 13],
                             [62, 19, 21, 35, 64, 46, 27, 10, 39, 17, 22, 34, 30],
                             [48, 15, 51, 6, 40, 31, 4, 36, 45, 41, 16, 14, 11],
                             [9, 29, 1, 44, 33, 26, 25, 32, 53, 12, 43, 38, 60],
                             [28, 2, 49, 37, 20, 23, 58, 24, 52, 61, 7, 0, 42] ]


#### Read Office-Home images in one domain among 'Art', 'Clipart', 'Product' and 'RealWorld'
def read_officehome_images(image_type, img_size):
    assert ( (image_type == 'Art') or (image_type == 'Clipart') or (image_type == 'Product') or (image_type == 'RealWorld') ), "Given image type of Office-Home dataset is wrong!"
    raw_images = []
    for class_label in _officehome_class_labels:
        img_path = os.getcwd()+'/Data/OfficeHomeDataset/'+image_type+'/'+class_label+'/'
        num_imgs = len(os.listdir(img_path))
        raw_class_images = np.zeros([num_imgs, img_size[0]*img_size[1]*img_size[2]], dtype=np.float32)
        for img_cnt in range(num_imgs):
            img_name = img_path + format(int(img_cnt+1), '05d') + '.jpg'
            img_tmp = mpimg.imread(img_name).astype(np.float32)/255.0
            img = skimage.transform.resize(img_tmp, img_size, anti_aliasing=True)
            raw_class_images[img_cnt] = np.array(img.reshape(-1))
        raw_images.append(np.array(raw_class_images))
        print("\t\tImage class - %s" %(class_label))
    return raw_images

#### Read Office-Home images in specified domains
def read_officehome_all_images(super_class_list, img_size):
    raw_images = {}
    if 'Art' in super_class_list:
        raw_images['Art'] = read_officehome_images('Art', img_size)
        print("\tComplete reading Office-Home/ Art images")
        print([c.shape[0] for c in raw_images['Art']])
        print("\n")

    if 'Clipart' in super_class_list:
        raw_images['Clipart'] = read_officehome_images('Clipart', img_size)
        print("\tComplete reading Office-Home/ Clipart images")
        print([c.shape[0] for c in raw_images['Clipart']])
        print("\n")

    if 'Product' in super_class_list:
        raw_images['Product'] = read_officehome_images('Product', img_size)
        print("\tComplete reading Office-Home/ Product images")
        print([c.shape[0] for c in raw_images['Product']])
        print("\n")

    if 'RealWorld' in super_class_list:
        raw_images['RealWorld'] = read_officehome_images('RealWorld', img_size)
        print("\tComplete reading Office-Home/ RealWorld images")
        print([c.shape[0] for c in raw_images['RealWorld']])
        print("\n")
    return raw_images


def officehome_data(data_file_name, num_train_ratio, num_valid_ratio, num_test_ratio, num_train_group, img_size):
    assert (num_train_ratio + num_valid_ratio + num_test_ratio <= 1.0), "Sum of the given ratio of data should be less than or equal to 1"

    curr_path = os.getcwd()
    if not ('Data' in os.listdir(curr_path)):
        os.mkdir('./Data')

    data_path = curr_path + '/Data'
    if data_file_name in os.listdir(data_path):
        with open('./Data/' + data_file_name, 'rb') as fobj:
            if sys.version_info.major < 3:
                train_data, validation_data, test_data = pickle.load(fobj)
            else:
                train_data, validation_data, test_data = pickle.load(fobj, encoding='latin1')
            print('Successfully load data')
    else:
        ## We used only 'Product' and 'RealWorld' domains.
        data_types = ['Product', 'RealWorld']
        raw_images = read_officehome_all_images(data_types, img_size)

        train_data, validation_data, test_data = [], [], []
        for group_cnt in range(num_train_group):
            train_data_tmp, validation_data_tmp, test_data_tmp = [], [], []

            for data_type in data_types:
                for task_cnt, (task_list) in enumerate(_officehome_task_labels):
                    for class_cnt, (class_label) in enumerate(task_list):
                        imgs_in_class_subtask = raw_images[data_type][class_label]
                        num_imgs = imgs_in_class_subtask.shape[0]
                        num_train, num_valid, num_test = int(num_imgs*num_train_ratio), int(num_imgs*num_valid_ratio), int(num_imgs*num_test_ratio)
                        img_indices = list(range(num_imgs))
                        shuffle(img_indices)

                        train_x_tmp, train_y_tmp = imgs_in_class_subtask[img_indices[0:num_train],:], np.ones(num_train, dtype=np.int32)*class_cnt
                        valid_x_tmp, valid_y_tmp = imgs_in_class_subtask[img_indices[num_train:num_train+num_valid],:], np.ones(num_valid, dtype=np.int32)*class_cnt
                        test_x_tmp, test_y_tmp = imgs_in_class_subtask[img_indices[num_train+num_valid:num_train+num_valid+num_test],:], np.ones(num_test, dtype=np.int32)*class_cnt

                        if class_cnt < 1:
                            train_x, train_y = train_x_tmp, train_y_tmp
                            valid_x, valid_y = valid_x_tmp, valid_y_tmp
                            test_x, test_y = test_x_tmp, test_y_tmp
                        else:
                            train_x, train_y = np.concatenate((train_x, train_x_tmp), axis=0), np.concatenate((train_y, train_y_tmp), axis=0)
                            valid_x, valid_y = np.concatenate((valid_x, valid_x_tmp), axis=0), np.concatenate((valid_y, valid_y_tmp), axis=0)
                            test_x, test_y = np.concatenate((test_x, test_x_tmp), axis=0), np.concatenate((test_y, test_y_tmp), axis=0)

                    train_x, train_y = shuffle_data_x_and_y(train_x, train_y)
                    valid_x, valid_y = shuffle_data_x_and_y(valid_x, valid_y)
                    test_x, test_y = shuffle_data_x_and_y(test_x, test_y)

                    train_data_tmp.append( ( np.array(train_x), np.array(train_y) ) )
                    validation_data_tmp.append( ( np.array(valid_x), np.array(valid_y) ) )
                    test_data_tmp.append( ( np.array(test_x), np.array(test_y) ) )

            train_data.append(train_data_tmp)
            validation_data.append(validation_data_tmp)
            test_data.append(test_data_tmp)
            num_tasks = len(train_data_tmp)

        #### save data
        with open('./Data/' + data_file_name, 'wb') as fobj:
            pickle.dump([train_data, validation_data, test_data], fobj)
            print('Successfully generate/save data')
    return (train_data, validation_data, test_data)

def officehome_data_print_info(train_data, valid_data, test_data, no_group=False, print_info=True):
    if no_group:
        num_task = len(train_data)

        num_train, num_valid, num_test = [train_data[x][0].shape[0] for x in range(num_task)], [valid_data[x][0].shape[0] for x in range(num_task)], [test_data[x][0].shape[0] for x in range(num_task)]
        x_dim, y_dim = train_data[0][0].shape[1], 0
        y_depth = int(np.amax(train_data[0][1])+1)
        if print_info:
            print("Tasks : %d\nTrain data : %d, Validation data : %d, Test data : %d" %(num_task, num_train, num_valid, num_test))
            print("Input dim : %d, Output dim : %d, Maximum label : %d\n" %(x_dim, y_dim, y_depth))
        return (num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth)
    else:
        assert (len(train_data) == len(valid_data) and len(train_data) == len(test_data)), "Different number of groups in train/validation data"
        num_group = len(train_data)

        bool_num_task = [(len(train_data[0]) == len(train_data[x])) for x in range(1, num_group)]
        assert all(bool_num_task), "Different number of tasks in some of groups in train data"
        bool_num_task = [(len(valid_data[0]) == len(valid_data[x])) for x in range(1, num_group)]
        assert all(bool_num_task), "Different number of tasks in some of groups in validation data"
        assert (len(train_data[0])==len(valid_data[0]) and len(train_data[0])==len(test_data[0])), "Different number of tasks in train/validation/test data"
        num_task = len(train_data[0])

        num_train, num_valid, num_test = [train_data[0][x][0].shape[0] for x in range(num_task)], [valid_data[0][x][0].shape[0] for x in range(num_task)], [test_data[0][x][0].shape[0] for x in range(num_task)]
        x_dim, y_dim = train_data[0][0][0].shape[1], 0
        y_depth = max([int(np.amax(train_data[0][x][1])) for x in range(num_task)])+1
        if print_info:
            print("Tasks : %d, Groups of training/valid : %d\n" %(num_task, num_group))
            print("Train data : ", num_train, ", Validation data : ", num_valid, ", Test data : ", num_test)
            print("Input dim : %d, Output dim : %d, Maximum label : %d\n" %(x_dim, y_dim, y_depth))
        return (num_task, num_group, num_train, num_valid, num_test, x_dim, y_dim, y_depth)


if __name__ == '__main__':

    ### CIFAR100 test
    '''
    data_type = 'cifar100'
    data_hyperpara = {}
    data_hyperpara['num_train_group'] = 5
    data_hyperpara['multi_class_label'] = True
    data_hyperpara['image_dimension'] = [32, 32, 3]
    data_hyperpara['num_train_data'], data_hyperpara['num_valid_data'], data_hyperpara['num_test_data'] = 1000, 100, 1000
    data_hyperpara['num_tasks'] = 10
    #data_file_name = data_type + '_mtl_data_group_' + str(data_hyperpara['num_train_data']) + '_' + str(data_hyperpara['num_valid_data']) + '_' + str(data_hyperpara['num_test_data']) + '_' + str(data_hyperpara['num_tasks']) + '.pkl'
    data_file_name = 'cifar100_test_delete_this.pkl'

    train_data, validation_data, test_data = cifar100_data(data_file_name, data_hyperpara['num_train_data'], data_hyperpara['num_valid_data'],
                                                           data_hyperpara['num_test_data'], data_hyperpara['num_train_group'],
                                                           data_hyperpara['num_tasks'], multiclass=data_hyperpara['multi_class_label'])
    cifar_data_print_info(train_data, validation_data, test_data)
    '''

    ### Office-Home
    tmp = officehome_data('temp.pkl', 0.6, 0.1, 0.3, 1, [128, 128, 3], save_as_mat=False)
    num_task, num_group, num_train, num_valid, num_test, x_dim, y_dim, y_depth = officehome_data_print_info(tmp[0], tmp[1], tmp[2], print_info=True)
