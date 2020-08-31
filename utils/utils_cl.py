import sys
import numpy as np
if sys.version_info.major < 3:
    from utils import convert_array_to_oneHot, compute_kernel_matrices, compute_information
else:
    from utils.utils import convert_array_to_oneHot, compute_kernel_matrices, compute_information


def compute_mutual_info_convNet(data, trained_model, model_hyperpara, train_hyperpara, num_task, batch_size, y_depth, sess):
    num_cnn_layers = len(model_hyperpara['kernel_sizes'])//2
    alpha, kernel_h, kernel_h_backward = train_hyperpara['mutual_info_alpha'], train_hyperpara['mutual_info_kernel_h'], train_hyperpara['mutual_info_kernel_h_backward']
    num_data = [data[x][0].shape[0] for x in range(num_task)]
    computed_forward_mutual_info, computed_backward_mutual_info = dict(), dict()

    for task_cnt in range(num_task):
        mutual_info_layers = [[] for _ in range(num_cnn_layers)]
        mutual_info_backward_layers = [[] for _ in range(num_cnn_layers)]
        for batch_cnt in range(num_data[task_cnt]//batch_size):
            data_x = data[task_cnt][0][batch_cnt*batch_size:(batch_cnt+1)*batch_size, :]
            data_y = convert_array_to_oneHot(data[task_cnt][1][batch_cnt*batch_size:(batch_cnt+1)*batch_size], y_depth)

            X_for_MI = np.reshape(data_x, [batch_size, -1, model_hyperpara['image_dimension'][-1]])
            Kernel_X = compute_kernel_matrices(X_for_MI, kernel_h)
            S_X = compute_information(alpha, kernel_h, kernel_x=Kernel_X)

            Y_for_MI = np.reshape(data_y, [batch_size, y_depth, 1])
            Kernel_Y = compute_kernel_matrices(Y_for_MI, kernel_h_backward)
            S_Y = compute_information(alpha, kernel_h_backward, kernel_x=Kernel_Y)

            for layer_cnt in range(num_cnn_layers):
                computed_features = sess.run(trained_model.test_models[task_cnt][layer_cnt], feed_dict={trained_model.model_input[task_cnt]: data_x, trained_model.dropout_prob: 1.0})
                F_for_MI = np.reshape(computed_features, [batch_size, -1, model_hyperpara['channel_sizes'][layer_cnt]])
                Kernel_F = compute_kernel_matrices(F_for_MI, kernel_h)
                S_F = compute_information(alpha, kernel_h, kernel_x=Kernel_F)
                S_XF = compute_information(alpha, kernel_h, kernel_x=Kernel_X*Kernel_F)
                mutual_info_layers[layer_cnt].append(S_X + S_F - S_XF)

                Kernel_F2 = compute_kernel_matrices(F_for_MI, kernel_h_backward)
                S_F2 = compute_information(alpha, kernel_h_backward, kernel_x=Kernel_F2)
                S_YF = compute_information(alpha, kernel_h_backward, kernel_x=Kernel_Y*Kernel_F2)
                mutual_info_backward_layers[layer_cnt].append(S_Y + S_F2 - S_YF)

        computed_forward_mutual_info['Task%d'%task_cnt] = np.array(mutual_info_layers)
        computed_backward_mutual_info['Task%d'%task_cnt] = np.array(mutual_info_backward_layers)
    return computed_forward_mutual_info, computed_backward_mutual_info