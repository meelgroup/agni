from __future__ import print_function


import os
import datetime
import sys
import time
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
import pickle
import copy
import random
import tensorflow.keras.backend as K
from numpy.core import multiarray
from collections import deque

import datetime

#######################
# USER SET PARAMETERS
#######################

model_filename = 'baseline_logistic_regression_model.hdf5'
testset_log_filename = 'baseline_logistic_regression_model_test_set_log.txt'
training_dataset_fire_only_directory = './preprocessed_data_fire_only/'
training_dataset_directory = './preprocessed_data/'
testset_directory = './preprocessed_data_2018_test_set/'

#######################

start_time = datetime.datetime.now()

# setting seed for reproducable results
np.random.seed(42)
tf.set_random_seed(42)

# setting the directory for fire only dataset
dataset_dir = training_dataset_fire_only_directory

# getting a list of all files in dataset_dir
dataset_filename_list = os.listdir(dataset_dir)
dataset_filepath_list = [dataset_dir + x for x in dataset_filename_list]

## added in fire data from 2017
# additional_dataset_dir = './preprocessed_data_fire_only_2017/'
# additional_dataset_filename_list = os.listdir(additional_dataset_dir)
# additional_dataset_filepath_list = [additional_dataset_dir + x for x in additional_dataset_filename_list]
# dataset_filepath_list = dataset_filepath_list + additional_dataset_filepath_list

# adding in non fire data
mostly_nonfire_dataset_dir = training_dataset_directory
mostly_nonfire_dataset_filename_list = os.listdir(mostly_nonfire_dataset_dir)
nonfire_filepath_list = [mostly_nonfire_dataset_dir + x for x in mostly_nonfire_dataset_filename_list]

# here we set 3 times so that the non fire data takes up 3/4 of the total training data
sampled_nonfire_filepath_list = np.random.choice(nonfire_filepath_list, 3 * len(dataset_filename_list), replace=False)
sampled_nonfire_filepath_list = list(sampled_nonfire_filepath_list)

dataset_filepath_list = dataset_filepath_list + sampled_nonfire_filepath_list

# defining neural network parameters
width = 32
height = 8
num_channel = 1

epoch_length = len(dataset_filename_list)

# create a function that returns cnn component of the model
def create_cnn_component():
    
    X_input = tf.keras.layers.Input(shape=(height, width, num_channel))
#     X = tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=3, activation='relu')(X_input)
    X = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=2, activation='relu')(X_input)
#     X = tf.keras.layers.Conv2D(32, kernel_size=(2, 2), strides=1, activation='relu')(X)
    X = tf.keras.layers.Flatten()(X)
    
    model = tf.keras.models.Model(X_input, X)
    return model

# creating a function that returns the network
def create_model():
    
    # cnn_comp = create_cnn_component()
    
    X_input = tf.keras.layers.Input(shape=(12, height, width, num_channel))
    
    X = tf.keras.layers.Flatten()(X_input)
#     X = tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=1, activation='relu')(X_input)
#     X = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1, activation='relu')(X)
#     X = tf.keras.layers.Conv2D(32, kernel_size=(2, 2), strides=1, activation='relu')(X)
    
    # X = tf.keras.layers.TimeDistributed(cnn_comp)(X_input)
    
#     X = tf.keras.layers.Flatten()(X)
    # X = tf.keras.layers.LSTM(64)(X)
    # dropping 30 percent of connections, using seed=42
    # X = tf.keras.layers.Dropout(rate=0.3, seed=42)(X)
    # X = tf.keras.layers.Dense(256, activation='relu')(X)
#     X = tf.keras.layers.Dense(128, activation='relu')(X)
    # X = tf.keras.layers.Dense(32, activation='relu')(X)
    # here we only have 1 final output, signifying the likelihood of fire
    X_out = tf.keras.layers.Dense(1, activation='sigmoid')(X)
    
    model = tf.keras.models.Model(X_input, X_out)
    return model

net = create_model()
# net.summary()

# defining custom loss function
def custom_loss(y_true,y_pred):
    # mean abs error implementation from keras
#     mae_tensor = K.mean(K.abs(y_pred - y_true), axis=-1)
    # +ve means real more than predicted, penalize
    # -ve means predicted more than real, normal loss
    diff_tensor = y_true - y_pred
    
    exponential_tensor = K.exp(diff_tensor * 100.0 / 30.0)
#     exponential_tensor = K.exp(diff_tensor * 100.0 / 20.0)
#     exponential_tensor = K.exp(diff_tensor * 100.0 / 11.0)
    exponential_tensor = K.clip(exponential_tensor, min_value=1.0, max_value=10000.0)
    abs_tensor = K.abs(diff_tensor)
    output_tensor = K.mean(abs_tensor * exponential_tensor)
#     clipped_tensor = K.clip(exponential_tensor, 1.0, 10000.0)
    return output_tensor


# setting learning rate
# opt = tf.keras.optimizers.Adam(lr=0.00001)

# net.compile(optimizer=opt, loss=custom_loss, metrics=['accuracy'])
net.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

def parse_input_data_function(filename):
    histogram_data, label = pickle.load(open(filename, 'rb'))
    histogram_data = histogram_data.transpose(1, 0)
    histogram_data = histogram_data.reshape([-1, 8, 32, 1])
    # we only want the most recent satellite image histogram to train the baseline
    # logistic regression model
    # histogram_data = histogram_data[:1, :, :, :]
    # we take the most recent 12 and flatten to train baseline logistic regression, because in the dataset instances
    # there are missing timesteps, and logistic regression cannot handle missing timesteps (aka missing features)
    histogram_data = histogram_data[:12, :, :, :]
    histogram_data = histogram_data[::-1, :, :, :]
#     data = tf.convert_to_tensor(histogram_data, dtype=tf.float32)
#     label = tf.convert_to_tensor(label, dtype=tf.float32)
    label = np.array(float(label) / 100.0)
    label = label.reshape(1, 1)
    return histogram_data.astype('float32'), label

# setting number of training epochs
# num_epoch = 20
num_epoch = 50

epoch_len = len(dataset_filepath_list)
dataset_files_list = copy.deepcopy(dataset_filepath_list)

# shuffle before starting
dataset_files_list = random.sample(dataset_files_list, len(dataset_files_list))

loss_history = deque(maxlen=500)
loss_history_overall = []

for i in range(num_epoch):
    # need shuffle
    print('Epoch:' +  str(i))
    dataset_files_list = random.sample(dataset_files_list, len(dataset_files_list))
    for x in range(epoch_len):
        current_file = dataset_files_list[x]
        input_data, label = parse_input_data_function(current_file)
        input_data = input_data.reshape([1, -1, 8, 32, 1])
        if x % 500 == 0:
            print('Current epoch progress:' + str(float(x) / epoch_len))
            print('Average loss for previous 500 instances: ' + str(np.mean(loss_history)))
            if not np.isnan(np.mean(loss_history)):
                loss_history_overall.append(np.mean(loss_history))
            result = net.fit(x=input_data, y=label, epochs=1)
        else:
            result = net.fit(x=input_data, y=label, epochs=1, verbose=0)
        loss_history.append(result.history['loss'])


net.save(model_filename)

training_completion_time = datetime.datetime.now()


# setting the directory for dataset
test_dataset_dir = testset_directory

# getting a list of all files in dataset_dir
test_data_files = os.listdir(test_dataset_dir)

real_label = []
model_prediction = []

test_data_filepath = [test_dataset_dir + x for x in test_data_files]

for i in range(len(test_data_filepath)):
    x_data, y_label = parse_input_data_function(test_data_filepath[i])
    x_data = x_data.reshape([1, -1, 8, 32, 1])
    current_prediction = net.predict(x_data)
    model_prediction.append(current_prediction[0])
    real_label.append(y_label)

prediction_array = np.array(model_prediction)
test_label_array = np.array(real_label)
test_label_array = test_label_array.reshape(-1, 1)

diff_array = test_label_array - prediction_array

log_file = open(testset_log_filename, 'a+')

log_file.write('Total number of instances in test set : ' + str(len(test_label_array)))
log_file.write('Number of fire in test set : ' + str(len(test_label_array[test_label_array > 0])))

# we are considering > 50% = fire
log_file.write('True positive number(fire and predicted fire) : ' + str(sum(prediction_array[test_label_array > 0] > 0.5)))
log_file.write('True negative number : ' + str(sum(prediction_array[test_label_array == 0] <= 0.5)))
log_file.write('False positive number : ' + str(sum(test_label_array[prediction_array > 0.5] == 0)))
log_file.write('False negative number : ' + str(sum(test_label_array[prediction_array <= 0.5] > 0)))

num_correct_predictions = sum(prediction_array[test_label_array > 0] > 0.5) + sum(prediction_array[test_label_array == 0] <= 0.2) * 1.0
num_wrong_predictions = sum(test_label_array[prediction_array > 0.5] == 0) + sum(test_label_array[prediction_array <= 0.5] > 0) * 1.0

log_file.write('Percentage correctly predicted : ' + str(num_correct_predictions / (num_correct_predictions + num_wrong_predictions)))

log_file.write('For fire instances, average difference(real label - prediction) : ' + str(np.mean(diff_array[test_label_array > 0])))
log_file.write('For non fire instances, average difference : ' + str(np.std(diff_array[test_label_array == 0])))

end_time = datetime.datetime.now()

log_file.write('Script start time : ' + str(start_time))
log_file.write('Training completion time : ' + str(training_completion_time))
log_file.write('Script end time : ' + str(end_time))

log_file.close()
