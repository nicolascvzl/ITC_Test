import numpy as np
import os
import csv
import pandas as pd
import xgboost as xgb
import pickle
from tools import *
from sklearn.model_selection import train_test_split

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Loading Datas
with open('data/dataset_dummy.pkl', 'rb') as f:
    train_data, test_data, labels, label_dict = pickle.load(f)

# Creating a train/val split
tmp_df = pd.concat([train_data, labels], axis=1)
train_set, val_set = train_test_split(tmp_df, test_size=0.2)
train_data = train_set.iloc[:, :-1]
train_labels = train_set.iloc[:, -1]
val_data = val_set.iloc[:, :-1]
val_labels = val_set.iloc[:, -1]

# Creation of the model
n_features = train_data.shape[1]
n_tokens = len(list(label_dict.keys()))

# Optimizing parameters with hand-made cross validation
search_params = {}
search_params['eta'] = [1, 0.5, 0.25, 0.1, 0.05, 0.01]
search_params['max_depth'] = [2, 5, 10, 15]
optimal_params = xgboost_parameter_optimization(train_data, labels, search_params)

# Training
xg_train = xgb.DMatrix(train_data.values, label=train_labels.values)

optimal_params['objective'] = 'multi:softmax'
optimal_params['silent'] = 0
optimal_params['nthread'] = 4
optimal_params['num_class'] = 8
optimal_params['eval_metric'] = 'auc'

print('Training the model...')
num_round = 25
bst = xgb.train(params=optimal_params, dtrain=xg_train, num_boost_round=num_round)
bst.save_model('data/xgboost_refactored_model')
print('Done')
print('-' * 50)

# Training accuracy
train_pred = bst.predict(xg_train)
train_accuracy = np.sum(train_pred == train_labels.values) / train_labels.values.shape[0]
print('Training Accuracy = ', train_accuracy * 100, '%')
print('-' * 50)

# Validation accuract
xg_val = xgb.DMatrix(val_data.values, label=val_labels.values)
val_pred = bst.predict(xg_val)
val_accuracy = np.sum(val_pred == val_labels.values) / val_labels.values.shape[0]
print('Validation Accuracy = ', val_accuracy * 100, '%')

# Solving the classification
xg_test = xgb.DMatrix(test_data.values)
pred = bst.predict(xg_test)
decoded_pred = np.array([searchin_dict(label_dict, pred[i]) for i in range(len(pred))])

# Dump the result in a csv file
with open('pred/xgboost_refactored_model_predictions.csv', 'w') as f:
    csv.writer(f).writerows(decoded_pred)
