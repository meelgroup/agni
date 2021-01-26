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

model_filename = 'model_9month.hdf5'
eval_set_log_filename = 'model_9month_eval_set_log_aug19.txt'
eval_set_directory = './preprocessed_data_2019_aug_with_sept_fire_test/'
prediction_output_file = 'prediction_label_9month_aug19.pickle'

#######################

model_file = model_filename

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

net = tf.keras.models.load_model(model_file, custom_objects={'custom_loss':custom_loss})

# test_dataset_dir = './preprocessed_data_2018_test_set/'
# test_dataset_dir = './preprocessed_data_2019_july_with_aug_fire_test/'
test_dataset_dir = eval_set_directory

testset_filename_list = os.listdir(test_dataset_dir)

testset_filepath_list = [test_dataset_dir + x for x in testset_filename_list]
testset_length = len(testset_filename_list)

def parse_input_data_function(filename):
    histogram_data, label = pickle.load(open(filename, 'rb'))
    histogram_data = histogram_data.transpose(1, 0)
    histogram_data = histogram_data.reshape([-1, 8, 32, 1])
    # at this point we only want the latest 18 instances which roughly corresponds to 9 months
    histogram_data = histogram_data[:18, :, :, :]
    histogram_data = histogram_data[::-1, :, :, :]
    histogram_data = histogram_data.reshape([1, -1, 8, 32, 1])
#     data = tf.convert_to_tensor(histogram_data, dtype=tf.float32)
#     label = tf.convert_to_tensor(label, dtype=tf.float32)
    label = np.array(float(label) / 100.0)
    label = label.reshape(1, 1)
    return histogram_data.astype('float32'), label

real_label = []
model_prediction = []

for i in range(testset_length):
    current_file = testset_filepath_list[i]
    input_data, label = parse_input_data_function(current_file)
    real_label.append(label)
    prediction = net.predict(input_data)
    model_prediction.append(prediction[0])

prediction_array = np.array(model_prediction)
test_label_array = np.array(real_label)
test_label_array = test_label_array.reshape(-1, 1)

pickle.dump((model_prediction, real_label), open(prediction_output_file, 'wb'))

diff_array = test_label_array - prediction_array

log_file = open(eval_set_log_filename, 'a+')

log_file.write('Total number of instances in test set : ' + str(len(test_label_array)) + '\n')
log_file.write('Number of fire in test set : ' + str(len(test_label_array[test_label_array > 0])) + '\n')

# we are considering > 50% = fire
log_file.write('True positive number(fire and predicted fire) : ' + str(sum(prediction_array[test_label_array > 0] > 0.5)) + '\n')
log_file.write('True negative number : ' + str(sum(prediction_array[test_label_array == 0] <= 0.5)) + '\n')
log_file.write('False positive number : ' + str(sum(test_label_array[prediction_array > 0.5] == 0)) + '\n')
log_file.write('False negative number : ' + str(sum(test_label_array[prediction_array <= 0.5] > 0)) + '\n')

num_correct_predictions = sum(prediction_array[test_label_array > 0] > 0.5) + sum(prediction_array[test_label_array == 0] <= 0.5) * 1.0
num_wrong_predictions = sum(test_label_array[prediction_array > 0.5] == 0) + sum(test_label_array[prediction_array <= 0.5] > 0) * 1.0

log_file.write('Percentage correctly predicted : ' + str(num_correct_predictions / (num_correct_predictions + num_wrong_predictions)) + '\n')

log_file.write('For fire instances, average difference(real label - prediction) : ' + str(np.mean(diff_array[test_label_array > 0])) + '\n')
log_file.write('For non fire instances, average difference : ' + str(np.std(diff_array[test_label_array == 0])) + '\n')

log_file.close()
