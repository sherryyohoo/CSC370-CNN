import numpy as np
import cv2
import tensorflow as tf
import os
sess = tf.Session()
batch_size = 10
generation = 3000
image_heigth = 512
image_width = 512
num_channels = 1
output_every = 50
eval_every = 50
learning_rate = 0.002
lr_decay = 0.9
num_gens_to_wait = 25
test_every = 500
#import data
data_set = np.zeros((110, 512, 512))
target_set = np.zeros((110, 512, 512))
boundary_set = np.zeros((110, 512, 512))
for i in range(0, 110):
    data_set[i] = np.float32(cv2.imread('train/image/'+str(i)+'.bmp', 0))
    target_set[i] = np.float32(cv2.imread('train/gtruth/'+str(i)+'.bmp', 0))
    boundary_set[i] = np.float32(cv2.imread('/train/boundary/'+str(i)+'.bmp', 0))
#create placeholder
train_input = tf.placeholder(dtype=tf.float32, shape=[batch_size, 512, 512, num_channels], name='train_input')
train_target = tf.placeholder(dtype=tf.float32, shape=[batch_size, 512, 512, num_channels], name='train_target')
test_input = tf.placeholder(dtype=tf.float32, shape=[1, 512, 512, num_channels], name='test_input')
test_target = tf.placeholder(dtype=tf.float32, shape=[1, 512, 512, num_channels], name='test_input')
boun_target = tf.placeholder(dtype=tf.float32, shape=[batch_size, 512, 512, num_channels], name='boun_target')
my_input = tf.placeholder(dtype=tf.uint8, shape=[1, 512, 512, num_channels], name='my_input')
# Define Loss function
def loss_function(logits, targets, boundary):
    lanmbda = tf.constant(5.0)
    targets = tf.squeeze(tf.cast(targets, tf.int32))
    labels = tf.one_hot(targets, 10)
    #logits = tf.cast(logits, tf.int32)
    #boundary = tf.cast(boundary, tf.int32)
    #boundur_targets = tf.one_hot(targets, 10)
    cross_entropy1 = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels)
    labels = tf.squeeze(tf.cast(labels, tf.float32))
    boundary = tf.squeeze(boundary)
    cross_entropy2 = tf.multiply(tf.reduce_mean(tf.square(logits - labels), 3), boundary) 
    cross_entropy_mean = tf.reduce_mean(cross_entropy1)+tf.reduce_mean(cross_entropy2)*lanmbda
    return(cross_entropy_mean)
# Define Unpooling function
def unpooling_function(pool_image, strides):
    shape_new = [pool_image.shape[0], pool_image.shape[1]*strides, pool_image.shape[2]*strides, pool_image.shape[3]]
    inference = tf.image.resize_nearest_neighbor(pool_image, tf.stack([shape_new[1], shape_new[2]]))
    return(inference)

# Define CNN model
def cnn_model(input_image, batch_size, train_logical = True):
    def truncated_normal_var(name, shape, dtype):
        return(tf.get_variable(name = name, shape = shape, dtype = dtype, initializer = tf.truncated_normal_initializer(stddev = 0.05)))
    def zero_var(name, shape, dtype):
        return(tf.get_variable(name = name, shape = shape, dtype = dtype, initializer = tf.truncated_normal_initializer(0.0)))
    input_image = tf.cast(input_image, tf.float32)
    # First convolution layer
    with tf.variable_scope('conv1') as scope:
        # Conv1 kernel is 7x3 and we will create 64 features
        #conv1_kernel = truncated_normal_var(name = 'conv_kernel1', shape = [7, 3, 1, 64], dtype = tf.float32)
        conv1_kernel = truncated_normal_var(name = 'conv_kernel1', shape = [7, 3, 1, 64], dtype = tf.float32)
        # We convolve the image with a strid of 1
        conv1 = tf.nn.conv2d(input_image, conv1_kernel, [1, 1, 1, 1], padding = 'SAME')
        # Initialize and add the bias term
        conv1_bias = zero_var(name = 'conv1_bias', shape = [64], dtype = tf.float32)
        conv1_add_bias = tf.nn.bias_add(conv1, conv1_bias)
        # ReLU element wise
        relu_conv1 = tf.nn.relu(conv1_add_bias)
    # Max pool layer1
    pool1 = tf.nn.max_pool(relu_conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'pool_layer1')
    # Local response normalization
    norm1 = tf.nn.lrn(pool1, depth_radius = 5, bias = 2.0, alpha = 1e-3, beta = 0.75, name = 'norm1')

    # Second convolution layer
    with tf.variable_scope('conv2') as scope:
        # Conv2 kernel is 7x3 and we will create 1 features
        conv2_kernel = truncated_normal_var(name = 'conv_kernel2', shape = [7, 3, 64, 64], dtype = tf.float32)
        # We convolve the image with a strid of 1
        conv2 = tf.nn.conv2d(norm1, conv2_kernel, [1, 1, 1, 1], padding = 'SAME')
        # Initialize and add the bias term
        conv2_bias = zero_var(name = 'conv2_bias', shape = [64], dtype = tf.float32)
        conv2_add_bias = tf.nn.bias_add(conv2, conv2_bias)
        # ReLU element wise
        relu_conv2 = tf.nn.relu(conv2_add_bias)
    # Max pool layer1
    pool2 = tf.nn.max_pool(relu_conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'pool_layer2')
    # Local response normalization
    norm2 = tf.nn.lrn(pool2, depth_radius = 5, bias = 2.0, alpha = 1e-3, beta = 0.75, name = 'norm2')

    # Third convolution layer
    with tf.variable_scope('conv3') as scope:
        # Conv3 kernel is 7x3 and we will create 1 features
        conv3_kernel = truncated_normal_var(name = 'conv_kernel3', shape = [7, 3, 64, 64], dtype = tf.float32)
        # We convolve the image with a strid of 1
        conv3 = tf.nn.conv2d(norm2, conv3_kernel, [1, 1, 1, 1], padding = 'SAME')
        # Initialize and add the bias term
        conv3_bias = zero_var(name = 'conv3_bias', shape = [64], dtype = tf.float32)
        conv3_add_bias = tf.nn.bias_add(conv3, conv3_bias)
        # ReLU element wise
        relu_conv3 = tf.nn.relu(conv3_add_bias)
    # Max pool layer1
    pool3 = tf.nn.max_pool(relu_conv3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'pool_layer3')
    # Local response normalization
    norm3 = tf.nn.lrn(pool3, depth_radius = 5, bias = 2.0, alpha = 1e-3, beta = 0.75, name = 'norm3')

    # Forth convolution layer
    with tf.variable_scope('conv4') as scope:
        # Conv4 kernel is 7x3 and we will create 1 features
        conv4_kernel = truncated_normal_var(name = 'conv_kernel4', shape = [7, 3, 64, 64], dtype = tf.float32)
        # We convolve the image with a strid of 1
        conv4 = tf.nn.conv2d(norm3, conv4_kernel, [1, 1, 1, 1], padding = 'SAME')
        # Initialize and add the bias term
        conv4_bias = zero_var(name = 'conv4_bias', shape = [64], dtype = tf.float32)
        conv4_add_bias = tf.nn.bias_add(conv4, conv4_bias)
        # ReLU element wise
        relu_conv4 = tf.nn.relu(conv4_add_bias)
    norm4 = tf.nn.lrn(relu_conv4, depth_radius = 5, bias = 2.0, alpha = 1e-3, beta = 0.75, name = 'norm4')
    # Unpool-Layer1
    unpool1 = tf.concat([unpooling_function(norm4, strides = 2), relu_conv3], 3)

    # Fifth convolution layer
    with tf.variable_scope('conv5') as scope:
        # Conv4 kernel is 7x3 and we will create 1 features
        conv5_kernel = truncated_normal_var(name = 'conv_kernel5', shape = [7, 3, 128, 64], dtype = tf.float32)
        # We convolve the image with a strid of 1
        conv5 = tf.nn.conv2d(unpool1, conv5_kernel, [1, 1, 1, 1], padding = 'SAME')
        # Initialize and add the bias term
        conv5_bias = zero_var(name = 'conv5_bias', shape = [64], dtype = tf.float32)
        conv5_add_bias = tf.nn.bias_add(conv5, conv5_bias)
        # ReLU element wise
        relu_conv5 = tf.nn.relu(conv5_add_bias)
    # Unpool_Layer2
    unpool2 = tf.concat([unpooling_function(relu_conv5, strides = 2), relu_conv2], 3)

    # Sixth convolution layer
    with tf.variable_scope('conv6') as scope:
        # Conv4 kernel is 7x3 and we will create 1 features
        conv6_kernel = truncated_normal_var(name = 'conv_kernel6', shape = [7, 3, 128, 64], dtype = tf.float32)
        # We convolve the image with a strid of 1
        conv6 = tf.nn.conv2d(unpool2, conv6_kernel, [1, 1, 1, 1], padding = 'SAME')
        # Initialize and add the bias term
        conv6_bias = zero_var(name = 'conv6_bias', shape = [64], dtype = tf.float32)
        conv6_add_bias = tf.nn.bias_add(conv6, conv6_bias)
        # ReLU element wise
        relu_conv6 = tf.nn.relu(conv6_add_bias)
    # Unpool-Layer3
    unpool3 = tf.concat([unpooling_function(relu_conv6, strides = 2), relu_conv1], 3)

    # Seventh convolution layer
    with tf.variable_scope('conv7') as scope:
        # Conv4 kernel is 7x3 and we will create 1 features
        conv7_kernel = truncated_normal_var(name = 'conv_kernel7', shape = [7, 3, 128, 64], dtype = tf.float32)
        # We convolve the image with a strid of 1
        conv7 = tf.nn.conv2d(unpool3, conv7_kernel, [1, 1, 1, 1], padding = 'SAME')
        # Initialize and add the bias term
        conv7_bias = zero_var(name = 'conv7_bias', shape = [64], dtype = tf.float32)
        conv7_add_bias = tf.nn.bias_add(conv7, conv7_bias)
        # ReLU element wise
        relu_conv7 = tf.nn.relu(conv7_add_bias)
    norm7 = tf.nn.lrn(relu_conv7, depth_radius = 5, bias = 2.0, alpha = 1e-3, beta = 0.75, name = 'norm7')

    # Eighth convolution layer
    with tf.variable_scope('conv8') as scope:
        conv8_kernel = truncated_normal_var(name = 'conv_kernel8', shape = [1, 1, 64, 10], dtype = tf.float32)
        conv8 = tf.nn.conv2d(norm7, conv8_kernel, [1, 1, 1, 1], padding = 'SAME')
        conv8_bias = zero_var(name = 'conv8_bais', shape = [10], dtype = tf.float32)
        conv8_add_bias = tf.nn.bias_add(conv8, conv8_bias)
    return(conv8_add_bias)

# Define learning rate descent function and realize Gradient descent
def train_step(loss_value, generations_num):
    model_learning_rate = tf.train.exponential_decay(learning_rate, generations_num, num_gens_to_wait, lr_decay, staircase = True)
    my_optimizer = tf.train.GradientDescentOptimizer(model_learning_rate)
    train_step = my_optimizer.minimize(loss_value)
    return(train_step)

# Def a accuracy function
def accuracy_of_batch(logits, targets):
    targets = tf.squeeze(tf.cast(targets, tf.int32))
    batch_predictions = tf.cast(tf.argmax(logits, 3), tf.int32)
    predicted_correctly = tf.equal(batch_predictions, targets)
    accuracy = tf.multiply(tf.reduce_mean(tf.cast(predicted_correctly, tf.float32)), 100.)
    return(accuracy)


######################################33
with tf.variable_scope('model_defination') as scope:
    model_output = cnn_model(train_input, batch_size)
    scope.reuse_variables()
    test_output = cnn_model(test_input, 1)
    my_output = cnn_model(my_input, 1)
loss = loss_function(model_output, train_target, boun_target)
accuracy = accuracy_of_batch(test_output, test_target)
generation_num = tf.Variable(0, trainable = False)
train_op = train_step(loss, generation_num)
init = tf.global_variables_initializer()
sess.run(init)
tf.train.start_queue_runners(sess = sess)
train_loss = []
test_accuracy = []
for i in range(generation):
    rand_index = np.random.choice((data_set.shape[0]), 11)
    rand_train_input = data_set[rand_index[0:10]].reshape((10, 512, 512, 1))
    rand_train_target = target_set[rand_index[0:10]].reshape((10, 512, 512, 1))
    rand_test_input = data_set[rand_index[-1]].reshape((1, 512, 512, 1))
    rand_test_target = target_set[rand_index[-1]].reshape((1, 512, 512, 1))
    rand_boudary_target = boundary_set[rand_index[0:10]].reshape((10, 512, 512, 1))
    _, loss_value = sess.run([train_op, loss], feed_dict={train_input:rand_train_input, train_target:rand_train_target, boun_target:rand_boudary_target})
    if (i+1) % output_every == 0:
        train_loss.append(loss_value)
        output = 'Generation {}: Loss = {};'.format((i+1), loss_value)
        print(output)
    if (i+1) % eval_every == 0:
        temp_accuracy = sess.run([accuracy], feed_dict={test_input:rand_test_input, test_target:rand_test_target})
        test_accuracy.append(temp_accuracy)
        acc_output = ' --- Test accuracy = {:.2f}%.'.format(temp_accuracy[0])
        print(acc_output)
    if (i+1)%test_every == 0:
        for j in range(7):
            image = cv2.imread('test/'+str(j)+'.jpeg', 0).reshape((1, 512, 512, 1))
            tem_image = sess.run(my_output, feed_dict={my_input:image})
            final_output = tf.squeeze(tf.arg_max(tem_image, 3))
            cv2.imwrite('graph/'+str(i+1)+'-'+str(j)+'.bmp', sess.run(final_output)*20)