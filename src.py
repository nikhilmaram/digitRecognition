from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here
## Improved Digit Recognition
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf

## Loading the data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

## Creating the Session
sess = tf.InteractiveSession()

## Building graph for Softmax Regression Model with multiple convolution layers

## To generate weight variable for differnent convolution layer
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

## TO generate bias variable for different convolution layer
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

## Input place holder
x = tf.placeholder(tf.float32, shape=[None, 784])
## Output
y_ = tf.placeholder(tf.float32, shape=[None, 10])

## Convolution function specifying stride as 1 in each dimension
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

## Max pool function with kernal size as 2*2 in the image(picks the maximum element in the 2*2 matrix) with stride length 2
## Since stride is 2, the output dimension decreases by 2 when compared to input dimension(since it skips a column after considering kernel)

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


## Building the first convolution layer with

## 1. Patch size for running convolution on the image as 5*5 and output of 32 features corresponding to each
## 2. Reshaping of image for convolution
## 3. Max-pooling the output of the convolution layer

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

## Resizing the image, the input vector is converted into multiple 28*28 vectors
x_image = tf.reshape(x, [-1, 28, 28, 1])

## RELU activation function is apploed, after a 2d convolution is done on initial image and filter chosen
## The values of the filter are learned over the period.
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
## max-pooling : this would result in 14*14 size of the image after max-pooling
h_pool1 = max_pool_2x2(h_conv1)

## Building the second convolution layer with
## 1. Patch size for running convolution on the image as 5*5 and output of 64 features corresponding to each
## 2. Reshaping of image for convolution
## 3. Max-pooling the output of the convolution layer

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

## max-pooling : this would result in 7*7 size of the image after max-pooling
h_pool2 = max_pool_2x2(h_conv2)

## Flattening the input to 1024
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

## Applying dropput method to avoid overfitting
## Place holder for the drop-out probability of a neuron
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## Final Read out layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

## Using cross entropy cost-function
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

## Using adam optimiser instead instead of Gradient Descent
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

## Evaluating the model

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

  print('test accuracy %g' % accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
