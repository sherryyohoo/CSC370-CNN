import numpy as np
import cv2
import tensorflow as tf

# load data
def data_load(path_data, path_truth, path_loss, num, heigth = 512, width = 512):
    '''
    inpput：
        path_data:  the training images
        path_truth: the target images
        path_loss:  the loss iamges
        num:        number of images for training
    return：
        three numpy arrays containing the data
    '''
    # create three numpy arrays for data storage
    data_set = np.zeros((num, heigth, width))
    data_truth = np.zeros((num, heigth, width))
    data_loss = np.zeros((num, heigth, width))
    for i in range(0, num):
        # load data information from grayscale images in dataset
        data_set[i] = cv2.imread(path_data + '/'+ str(i) + '.bmp', 0)
        data_truth[i] = cv2.imread(path_truth + '/'+ str(i) + '.bmp', 0)
        data_loss[i] = cv2.imread(path_loss + '/'+ str(i) + '.bmp', 0)
    return(data_set, data_truth, data_loss)

#  define the loss function
def loss_function(logits, targets, boundary):
    '''
    input：
        logits:   input images
        targets:  the target images coresponding to the input images
        boundary: for the boundary information
    return：
        cross_entropy_mean： loss
    '''
    lanmbda = tf.constant(5.0)
    targets = tf.squeeze(tf.cast(targets, tf.int32))
    labels = tf.one_hot(targets, 10)
    # cross_entropy
    cross_entropy1 = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels)
    labels = tf.squeeze(tf.cast(labels, tf.float32))
    boundary = tf.cast(tf.squeeze(boundary), tf.float32)
    # calculate boundary penalty
    cross_entropy2 = tf.multiply(tf.reduce_mean(tf.square(logits - labels), 3), boundary) 
    # calculate the composite loss
    cross_entropy_mean = tf.reduce_mean(cross_entropy1)+tf.reduce_mean(cross_entropy2)*lanmbda
    return(cross_entropy_mean)

# define the unpooling function
def unpooling_function(pool_image, strides):
    '''
    input：
        pool_image: the image after pooling
        strides:    strides
    return：
        inference:  image after unpooling
    '''
    shape_new = [pool_image.shape[0], pool_image.shape[1]*strides, pool_image.shape[2]*strides, pool_image.shape[3]]
    inference = tf.image.resize_nearest_neighbor(pool_image, tf.stack([shape_new[1], shape_new[2]]))
    return(inference)

# Define CNN model
def cnn_model(input_image, batch_size, train_logical = True):
    '''
    input：
        imput_image:  input size
        batch_size:   batch size
    return：
        conv8_add_bias： cnn model
    '''
    def truncated_normal_var(name, shape, dtype):
        return(tf.get_variable(name = name, shape = shape, dtype = dtype, initializer = tf.truncated_normal_initializer(stddev = 0.05)))
    def zero_var(name, shape, dtype):
        return(tf.get_variable(name = name, shape = shape, dtype = dtype, initializer = tf.truncated_normal_initializer(0.0)))
    input_image = tf.cast(input_image, tf.float32)
    # First convolution layer
    with tf.variable_scope('conv1') as scope:
        # Conv1 kernel is 7x3 and we will create 64 features
        conv1_kernel = truncated_normal_var(name = 'conv_kernel1', shape = [7, 3, 1, 64], dtype = tf.float32)
        # We convolve the image with a stride of 1
        conv1 = tf.nn.conv2d(input_image, conv1_kernel, [1, 1, 1, 1], padding = 'SAME')
        # Initialize and add the bias term
        conv1_bias = zero_var(name = 'conv1_bias', shape = [64], dtype = tf.float32)
        conv1_add_bias = tf.nn.bias_add(conv1, conv1_bias)
        # ReLU
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
        # ReLU
        relu_conv2 = tf.nn.relu(conv2_add_bias)
    # Max pool layer2
    pool2 = tf.nn.max_pool(relu_conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'pool_layer2')
    # Local response normalization
    norm2 = tf.nn.lrn(pool2, depth_radius = 5, bias = 2.0, alpha = 1e-3, beta = 0.75, name = 'norm2')

    # Third convolution layer
    with tf.variable_scope('conv3') as scope:
        # Conv3 kernel is 7x3 and we will create 64 features
        conv3_kernel = truncated_normal_var(name = 'conv_kernel3', shape = [7, 3, 64, 64], dtype = tf.float32)
        # We convolve the image with a stride of 1
        conv3 = tf.nn.conv2d(norm2, conv3_kernel, [1, 1, 1, 1], padding = 'SAME')
        # Initialize and add the bias term
        conv3_bias = zero_var(name = 'conv3_bias', shape = [64], dtype = tf.float32)
        conv3_add_bias = tf.nn.bias_add(conv3, conv3_bias)
        # ReLU
        relu_conv3 = tf.nn.relu(conv3_add_bias)
    # Max pool layer3
    pool3 = tf.nn.max_pool(relu_conv3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'pool_layer3')
    # Local response normalization
    norm3 = tf.nn.lrn(pool3, depth_radius = 5, bias = 2.0, alpha = 1e-3, beta = 0.75, name = 'norm3')

    # Forth convolution layer
    with tf.variable_scope('conv4') as scope:
        # Conv4 kernel is 7x3 and we will create 64 features
        conv4_kernel = truncated_normal_var(name = 'conv_kernel4', shape = [7, 3, 64, 64], dtype = tf.float32)
        # We convolve the image with a strid of 1
        conv4 = tf.nn.conv2d(norm3, conv4_kernel, [1, 1, 1, 1], padding = 'SAME')
        # Initialize and add the bias term
        conv4_bias = zero_var(name = 'conv4_bias', shape = [64], dtype = tf.float32)
        conv4_add_bias = tf.nn.bias_add(conv4, conv4_bias)
        # ReLU
        relu_conv4 = tf.nn.relu(conv4_add_bias)
    norm4 = tf.nn.lrn(relu_conv4, depth_radius = 5, bias = 2.0, alpha = 1e-3, beta = 0.75, name = 'norm4')
    # Unpool-Layer1
    unpool1 = tf.concat([unpooling_function(norm4, strides = 2), relu_conv3], 3)

    # Fifth convolution layer
    with tf.variable_scope('conv5') as scope:
        # Conv5 kernel is 7x3 and we will create 64 features
        conv5_kernel = truncated_normal_var(name = 'conv_kernel5', shape = [7, 3, 128, 64], dtype = tf.float32)
        # We convolve the image with a strid of 1
        conv5 = tf.nn.conv2d(unpool1, conv5_kernel, [1, 1, 1, 1], padding = 'SAME')
        # Initialize and add the bias term
        conv5_bias = zero_var(name = 'conv5_bias', shape = [64], dtype = tf.float32)
        conv5_add_bias = tf.nn.bias_add(conv5, conv5_bias)
        # ReLU
        relu_conv5 = tf.nn.relu(conv5_add_bias)
    # Unpool_Layer2
    unpool2 = tf.concat([unpooling_function(relu_conv5, strides = 2), relu_conv2], 3)

    # Sixth convolution layer
    with tf.variable_scope('conv6') as scope:
        # Conv6 kernel is 7x3 and we will create 64 features
        conv6_kernel = truncated_normal_var(name = 'conv_kernel6', shape = [7, 3, 128, 64], dtype = tf.float32)
        # We convolve the image with a strid of 1
        conv6 = tf.nn.conv2d(unpool2, conv6_kernel, [1, 1, 1, 1], padding = 'SAME')
        # Initialize and add the bias term
        conv6_bias = zero_var(name = 'conv6_bias', shape = [64], dtype = tf.float32)
        conv6_add_bias = tf.nn.bias_add(conv6, conv6_bias)
        # ReLU
        relu_conv6 = tf.nn.relu(conv6_add_bias)
    # Unpool-Layer3
    unpool3 = tf.concat([unpooling_function(relu_conv6, strides = 2), relu_conv1], 3)

    # Seventh convolution layer
    with tf.variable_scope('conv7') as scope:
        # Conv7 kernel is 7x3 and we will create 64 features
        conv7_kernel = truncated_normal_var(name = 'conv_kernel7', shape = [7, 3, 128, 64], dtype = tf.float32)
        # We convolve the image with a strid of 1
        conv7 = tf.nn.conv2d(unpool3, conv7_kernel, [1, 1, 1, 1], padding = 'SAME')
        # Initialize and add the bias term
        conv7_bias = zero_var(name = 'conv7_bias', shape = [64], dtype = tf.float32)
        conv7_add_bias = tf.nn.bias_add(conv7, conv7_bias)
        # ReLU
        relu_conv7 = tf.nn.relu(conv7_add_bias)
    norm7 = tf.nn.lrn(relu_conv7, depth_radius = 5, bias = 2.0, alpha = 1e-3, beta = 0.75, name = 'norm7')

    # Eighth convolution layer (classification block)
    with tf.variable_scope('conv8') as scope:
        conv8_kernel = truncated_normal_var(name = 'conv_kernel8', shape = [1, 1, 64, 10], dtype = tf.float32)
        conv8 = tf.nn.conv2d(norm7, conv8_kernel, [1, 1, 1, 1], padding = 'SAME')
        conv8_bias = zero_var(name = 'conv8_bais', shape = [10], dtype = tf.float32)
        conv8_add_bias = tf.nn.bias_add(conv8, conv8_bias)
    return(conv8_add_bias)

# define the train step
def train_step(loss_value, generations_num, learning_rate, num_gens_to_wait, lr_decay):
    '''
    input：
        loss_value:       loss
        generations_num:  interation times
        learning_rate:    initial learning rate
        num_gens_to_wait： decay steps
        lr_decay:         decay rate, must be less than 1
    '''

    model_learning_rate = tf.train.exponential_decay(learning_rate, generations_num, num_gens_to_wait, lr_decay, staircase = True)
    my_optimizer = tf.train.GradientDescentOptimizer(model_learning_rate)
    train_step = my_optimizer.minimize(loss_value)
    return(train_step)

# define the accuracy rate
def accuracy_of_batch(logits, targets):
    '''
    input：
        logits:  input image
        targets: the target image coresponding with the input
    return：
        accuracy: accuracy rate
    '''
    targets = tf.squeeze(tf.cast(targets, tf.int32))
    batch_predictions = tf.cast(tf.argmax(logits, 3), tf.int32)
    predicted_correctly = tf.equal(batch_predictions, targets)
    accuracy = tf.multiply(tf.reduce_mean(tf.cast(predicted_correctly, tf.float32)), 100.)
    return(accuracy)
