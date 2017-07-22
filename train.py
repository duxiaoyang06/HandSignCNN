#import matplotlib.pyplot as plt
from __future__ import print_function
import tensorflow as tf
import pandas as pd
import numpy as np
#from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import dataset
import random
import math as math
import os
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes



# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.



IMAGE_WIDTH   = 52
IMAGE_HEIGHT  = 52

# Number of color channels for the images: 1 channel for gray-scale.
num_channels = 1

# image dimensions (only squares for now)
img_size = 52

# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# class info

classes = ['A', 'B','C']
num_classes = len(classes)

# batch size
batch_size = 100

# validation split
validation_size = .2

# how long to wait after validation loss stops improving before terminating training
early_stopping = None  # use None if you don't want to implement early stoping


train_path='training_data'
test_path='testing_data'


data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)
test_images = dataset.read_test_set(test_path, img_size,classes)
#print("ASdr")
#print(test_images.test._images.shape)
#xbatch_test = test_images.test._images
#xbatch_test = xbatch_test.reshape(41, img_size_flat) #here


print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(test_images.test.labels)))
print("- Validation-set:\t{}".format(len(data.valid.labels)))



def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



session = tf.Session()
#x = tf.placeholder(tf.float32, shape=[None,784])
#Tensor("Placeholder:0", shape=(?, 10), dtype=float32)
#training label
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)
#Tensor("Placeholder_1:0", shape=(?, 52, 52, 3), dtype=float32)
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

W_conv1 = weight_variable([5,5,num_channels,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
#batch norm
norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(norm1,W_conv2)+b_conv2)
#batch norm
norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
h_pool2 = max_pool_2x2(norm2)


W_fc1 = weight_variable([13*13*64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,13*13*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

keep_prob = tf.placeholder(tf.float32,name='keep_prob')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024,num_classes])
b_fc2 = bias_variable([num_classes])

#prediccion de label, cada posicion es un numero, el vector nO esta normalizado
#Tensor("add_3:0", shape=(?, 10), dtype=float32)
y_conv = tf.matmul(h_fc1_drop,W_fc2)+b_fc2
y_conv = tf.identity(y_conv, name="y_conv")
y_conv_cls = tf.argmax(y_conv, dimension=1)
y_conv_cls = tf.identity(y_conv_cls, name="y_conv_cls")

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#compara maximos indices entre vector predictivo y label en el training
#print("XXX")
#print(y_conv.shape)
#print(y_conv_cls.shape)
#print(y_true_cls.shape)
#(?, 3)
#(?,)
#(?,)
#correct_prediction = tf.equal(tf.argmax(y_conv_cls,1),tf.argmax(y_true_cls,1))

correct_prediction = tf.equal(y_conv_cls,y_true_cls)
#accuracy sobre todos los samples
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

session.run(tf.initialize_all_variables()) # for older versions
train_batch_size = batch_size

def print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    # Calculate the accuracy on the training-set.
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    #mine
    #test_acc = session.run(accuracy, feed_dict=feed_dict_test)
    msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))
    #print("y true")
    #print(session.run(y_true,feed_dict=feed_dict_test))
    #print("y predict")
    #print(session.run(y_conv,feed_dict=feed_dict_test))
    return acc,val_acc


total_iterations = 0

def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    best_val_loss = float("inf")

    for i in range(total_iterations,total_iterations + num_iterations):
        #print("total iterations:")
        #print(i)
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(train_batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(train_batch_size)
        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, flattened image shape]
        #x_batch = x_batch.reshape(train_batch_size, img_size_flat)
        x_batch = x_batch.reshape(train_batch_size, img_size_flat)
        #[None,IMAGE_HEIGHT,IMAGE_WIDTH,3]
        #x_valid_batch = x_valid_batch.reshape(train_batch_size, img_size_flat)
        x_valid_batch = x_valid_batch.reshape(train_batch_size, img_size_flat)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch,
                           keep_prob:0.5}
        
        feed_dict_validate = {x: x_valid_batch,
                              y_true: y_valid_batch,
                              keep_prob:1}
                              
        #feed_dict_test = {x: xbatch_test,
        #                      y_true: test_images.test._labels,
        #                      keep_prob:1}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(train_step, feed_dict=feed_dict_train)
        # Print status at end of each epoch (defined as full pass through training dataset).
        #if i % int(data.train.num_examples/batch_size) == 0: 
        #    val_loss = session.run(cross_entropy, feed_dict=feed_dict_validate)
        #    epoch = int(i / int(data.train.num_examples/batch_size))
        #    print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss)
        if i % 10 == 0:
            val_loss = session.run(cross_entropy, feed_dict=feed_dict_validate)
            epoch = int(i / int(data.train.num_examples/batch_size))
            train_acc,val_acc = print_progress(epoch, feed_dict_train, feed_dict_validate,val_loss)
            if(val_acc > 0.96 and val_acc>0.98 and val_loss<0.1):
                print("saving snapshot...")
                saver = tf.train.Saver()
                saver.save(session, 'my_test_model_iteration_'+str(i)) 
                saver.save(session, './my_test_model_iteration_'+str(i))
        if i % 100 == 0:
            print("total iterations:")
            print(i)
            val_loss = session.run(cross_entropy, feed_dict=feed_dict_validate)
            epoch = int(i / int(data.train.num_examples/batch_size))
            train_acc,val_acc = print_progress(epoch, feed_dict_train, feed_dict_validate,val_loss)
            if(val_acc > 0.96 and val_acc>0.98 and val_loss<0.1):
                print("saving snapshot...")
                saver = tf.train.Saver()
                saver.save(session, 'my_test_model_iteration_'+str(i)) 
                saver.save(session, './my_test_model_iteration_'+str(i)) 

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    
optimize(num_iterations=3000)
#print_validation_accuracy()
