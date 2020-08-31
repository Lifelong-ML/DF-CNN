# Sharing Less is More: Lifelong Learning in Deep Networks with Selective Layer Transfer
This is code of Lifelong Machine Learning workshop paper at ICML 2020. This work consists of two parts: (1) the effect of transfer configuration on the existing lifelong learning architectures and (2) the proposed EM-based selective transfer algorithm (Lifelong Architecture Search via EM: LASEM).


## Version and Dependencies
- Python 3.6 or higher
- numpy, scipy, scikit-image, matplotlib
- TensorFlow == 1.14

## Data
- CIFAR-100 (Lifelong)
    - Dataset consists of images of 100 classes.
    - Each task is 10-class classification task, and there are 10 tasks for the lifelong learning task with heterogeneous task distribution (disjoint set of image classes for these sub-tasks).
    - We trained models by using only 4% of the available dataset.
    - We normalized images.

- Office-Home (Lifelong)
    - We used images in Product and Real-World domains.
    - Each task is 13-class classification task, and image classes of sub-tasks are randomly chosen without repetition (but distinguishing classes from Product domain and those from Real-World domain).
    - Images are rescaled to 128x128 size and rescaled range of pixel value to [0, 1], but not normalized or augmented.

- STL-10 (Lifelong)
    - STL-10 dataset consists of images of 10 classes with resolution of 96x96.
    - We generated 20 tasks of three-way classification randomly. (random selection on image classes, mean and variance of Gaussian noise, and the order of channel permutation for the task)
    - We rescaled range of image value to [-0.5, 0.5].

## Implemented Model
- Hard-parameter Shared model
    - Neural networks for tasks share convolution layers, and have independent fully-connected layers for output.

- [Tensor Factorization](https://arxiv.org/abs/1605.06391) [model](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-BulatA.1460.pdf)
    - Factorize parameter of each layer into multiplication of several tensors, and share one tensor over different tasks as knowledge base. (Details in the paper Bulat, Adrian, Jean Kossaifi, Georgios Tzimiropoulos, and Maja Pantic. "Incremental multi-domain learning with network latent tensor factorization." ICML (2020).)

- [Dynamically Expandable Network model](https://arxiv.org/abs/1708.01547)
    - Extended hard-parameter shared model by retraining some neurons selectively/adding new neurons/splitting neurons into disjoint groups for different set of tasks according to the given data.
    - The code (cnn_den_model.py) is almost the same as code provided by authors.

- [Progressive Neural Network](https://arxiv.org/abs/1606.04671)
    - Introduce layer-wise lateral connections from earlier tasks to the current task to allow one-way transfer of knowledge.
    
- [Differentiable Architecture Search (DARTS)](https://arxiv.org/abs/1806.09055)
    - Neural Architecture Search method which is applicable to selective layer transfer.

- [Deconvolutional Factorized CNN](https://www.ijcai.org/Proceedings/2019/393)
    - Lifelong learning architecture consists of shared knowledge base and task-specific mappings based on deconvolutional operation.
    - The code (cnn_dfcnn_model.py) is almost the same as code provided by authors.
    
- Lifelong Learning Model with Static Transfer Configurations
    - This applies to hard-parameter sharing, tensor factorization and DF-CNN model.
    - One specific transfer configuration is used for all task models.

- Our Proposed LASEM (Lifelong Architecture Search via EM)
    - LASEM is applied to hard-parameter sharing, tensor factorization and DF-CNN model, which can be found in classification/model/lifelong_ver/cnn_lasem_model.py.

## Setting to run codes
1. Download data (CIFAR100, Office-Home and/or STL-10) from the original repositories, and place files under 'Data' directory after unzipping as follows:
    - CIFAR100 : Data/cifar-100-python (e.g. Data/cifar100-python/train)
    - Office-Home : Data/OfficeHomeDataset (e.g. Data/OfficeHomeDataset/Art/Alarm_Clock)
    - STL-10 : Data/STL-10/raw_data (e.g. Data/STL-10/raw_data/train_X.bin)
   
2. Install dependencies

3. Use below commands to run code, especially main_train_cl.py

**Caution!** Implementation of dynamically expandable network highly depends on the authors' code obtained by direct contact to the authors (not public code). Hence, we exclude DEN implementation from these codes. 

## Sample command to train a specific model
1. Hard-parameter Sharing with Top-2 transfer configuration on CIFAR100 (check line 478 of utils/utils_env_cl.py)

    ```python3 main_train_cl.py --gpu 0 --data_type CIFAR100_10 --data_percent 4 --num_clayers 4 --model_type HPS --test_type 10 --lifelong --save_mat_name cifar100_hpsTop2.mat```

2. Tensor Factorization with Bottom-3 transfer configuration on Office-Home (check line 478 of utils/utils_env_cl.py)

    ```python3 main_train_cl.py --gpu 0 --data_type OfficeHome --num_clayers 4 --model_type Hybrid_TF --test_type 30 --lifelong --save_mat_name officehome_tfBottom3.mat```
    
3. Deconvolutional Factorized CNN with Alter. transfer configuration on STL-10 (check line 478 of utils/utils_env_cl.py)

    ```python3 main_train_cl.py --gpu 0 --data_type STL10_20t --num_clayers 6 --model_type Hybrid_DFCNN --test_type 55 --lifelong --save_mat_name stl10_dfcnnAlter.mat```

4. LASEM HPS/TF/DFCNN on CIFAR100 (change model_type argument to either hybrid_hps_em/hybrid_tf_em/hybrid_dfcnn_auto_em)

    ```python3 main_train_cl.py --gpu 0 --data_type CIFAR100_10 --data_percent 4 --num_clayers 4 --model_type Hybrid_HPS_EM --lifelong --save_mat_name cifar100_lasem_hps.mat```
    
5. Dynamically Expandable Net on Office-Home

    ```python3 main_train_cl.py --gpu 0 --data_type OfficeHome --num_clayers 4 --model_type DEN --lifelong --save_mat_name officehome_den.mat```
    
6. Progressive Neural Net on Office-Home

    ```python3 main_train_cl.py --gpu 0 --data_type OfficeHome --num_clayers 4 --model_type ProgNN --test_type 1 --lifelong --save_mat_name officehome_prognn.mat```
    
7. Differentiable Architecture Search (DARTS) HPS/DFCNN on Office-Home (change model_type argument to either hybrid_hps_darts/hybrid_dfcnn_darts)

    ```python3 main_train_cl.py --gpu 0 --data_type OfficeHome --num_clayers 4 --model_type Hybrid_HPS_DARTS --lifelong --save_mat_name officehome_darts_hps.mat```


## (Miscellaneous) Code Files
1. main_train_cl.py
    main function to set-up model and data, and run training for classification tasks.

2. Directory *classification*
    This directory contains codes for classification tasks.
    - gen_data.py
        generate or load dataset, and convert its format for MTL/LL experiment
    - train_wrapper.py
        wrapper of train function in train.py to run independent training trials and save statistics into .mat file
    - train.py
        actual training functions for (simple) lifelong learning architectures exist.
    - train_selective_transfer.py
        training functions for selective transfer methods (LASEM and DARTS) exist.
    - Directory *model*
        This directory contains codes of neural network models.

3. Directory *utils*
    This directory contains utility functions.
    - utils_env_cl.py
        every hyper-parameter related to dataset and neural net models are defined.
    - utils.py
        functions which are miscellaneous but usable in general situation are defined. (e.g. handler of placeholder and output of MTL/LL models)
    - utils_cl.py 
        miscellaneous functions for classification tasks are defined.
    - utils_nn.py and utils_df_nn.py
        functions directly related to the construction of neural networks are defined. (e.g. function to generate convolutional layer)
