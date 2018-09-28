import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from copy import deepcopy

def matrix_to_one_hot(dictionary, idx_matrix):
    total_idx = len(list(dictionary.keys()))
    output = np.zeros([len(idx_matrix), total_idx])
    for i in range(len(idx_matrix)):
        output[i, idx_matrix[i]] = 1
    return output


def searchin_dict(dictionary, value):
    for key in dictionary:
        if dictionary[key] == value:
            return key
    return -1


def plot_model_history(model_history, file_name):
    fig, axs = plt.subplots(1,2,figsize=(15,5))

    # Summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')

    # Summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()
    plt.savefig('fig/'+ file_name + '.png')


def split_set(dataset, labels, val_size):
    """
    :param dataset: pd.DataFrame (features DataFrame)
    :param labels: pd.DataFrame (labels DataFrame)
    :param val_size: len(val_set) = len(train_set) // val_size
    :return: train_set: DMatrix(with labels), val_set: DMatrix(with labels), train_labels: np.array, val_labels: np.arry
    """
    df = pd.concat([dataset, labels], axis=1)
    train_set, val_set = train_test_split(df, test_size=val_size)

    train_data = train_set.iloc[:, :-1]
    train_labels = train_set.iloc[:, -1]

    val_data = val_set.iloc[:, :-1]
    val_labels = val_set.iloc[:, -1]

    train_set = xgb.DMatrix(train_data.values, label=train_labels.values)
    val_set = xgb.DMatrix(val_data.values, label=val_labels.values)

    return train_set, train_labels.values, val_set, val_labels.values


def xgboost_parameter_optimization(dataset, labels, parameter_dictionary):
    """
    :param dataset: pd.DataFrame of the training set
    :param labels: pd.DataFrame of the labels
    :param parameter_dictionary: of the form {param_1:[list of possible values], ...}
    :return: a dictionary of optimal parameters of the form {param_1: optimal_param_1, ...}
    """
    optimal_parameters = {'objective': 'multi:softmax', 'silent':1, 'nthread': 4, 'num_class': 8, 'eval_metric': 'auc'}
    num_round = 5

    for param in parameter_dictionary:
        best_val_acc = 0.
        best_param_value = None

        print('Optimizing ', param, '...')
        for param_value in tqdm(parameter_dictionary[param]):
            tmp_params = deepcopy(optimal_parameters)
            tmp_params[param] = param_value
            curr_acc = 0.

            for i in range(3):
                # Training temporary model
                xg_train, _, xg_val, val_labels = split_set(dataset, labels, 0.2)
                tmp_model = xgb.train(params=tmp_params, dtrain=xg_train, num_boost_round=num_round)

                # Evaluation temporary model
                val_pred = tmp_model.predict(xg_val)
                curr_acc += np.sum(val_pred == val_labels) / val_labels.shape[0]

            mean_val_acc = curr_acc / 3
            if mean_val_acc > best_val_acc:
                best_param_value = param_value
                best_val_acc = mean_val_acc
                # TODO: erase debugging print
                print('Param ', param, ' with value ', best_param_value, ' just beat the best val_acc, scoring: ', mean_val_acc * 100, '%')

        optimal_parameters[param] = best_param_value
        print('Best value for ', param, ': ', best_param_value)
        print('Done')
        print('-' * 50)

    return optimal_parameters
