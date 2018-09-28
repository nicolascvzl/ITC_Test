import csv
from tools import *
import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import numpy as np
import pickle

# Loading data
with open('data/dataset_dummy.pkl', 'rb') as f:
    train_data, test_data, labels, label_dict = pickle.load(f)

# Converting labels to one_hot
train_data = train_data.values
train_labels = labels.values
train_labels = matrix_to_one_hot(label_dict, train_labels)

test_data = test_data.values

# Creation of the model
n_features = train_data.shape[1]
n_tokens = len(list(label_dict.keys()))
dropout = 0.5
num_layers = 6

print('Defining model...')
model = Sequential()
model_name = 'neural_network_dummy_' + 'dropout_' + str(dropout) + '_n_layers_' + str(num_layers)


# Sequentially adding linear layers with a relu activation layer with dropout to avoid specialization of hidden units, in a bottleneck architecture
# Uniformly initializing weights to avoid weights to remain the same along training

model.add(Dense(256, input_dim=n_features, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(dropout))

model.add(Dense(128, input_dim=n_features, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(dropout))

model.add(Dense(64, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(dropout))

model.add(Dense(32, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(dropout))

model.add(Dense(16, kernel_initializer='uniform', activation='relu'))

model.add(Dense(n_tokens, activation='softmax', kernel_initializer='uniform'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
print('Done')
print('-' * 50)

# Defining early stopping callback to avoid over fitting
earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, verbose=1, mode='auto')
callbacks = [earlystop]

# Fitting the model to the data (learning) and saving the resulting model
print('Training model...')
model_info = model.fit(train_data, train_labels, epochs=100, batch_size=50, callbacks=callbacks, validation_split=0.2)

model.save('models/' + model_name + '.h5')
print('Done')
print('-' * 50)

plot_model_history(model_history=model_info, file_name=model_name)

# Solving the classification problem
output = model.predict(test_data)
pred = np.argmax(output, axis=1)
decoded_pred = np.array([searchin_dict(label_dict, pred[i]) for i in range(len(pred))])
out = np.column_stack((pred, decoded_pred))

with open('pred/' + model_name + '_predictions.csv', 'w') as f:
    csv.writer(f).writerows(decoded_pred)