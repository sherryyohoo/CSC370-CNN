import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from function import data_load
from function import loss_function
from function import unpooling_function
from function import cnn_model
from function import train_step
from function import accuracy_of_batch
sess = tf.Session()
# ---------------------#
# define the size of batch
batch_size = 10

#define image size
image_heigth = 512
image_width = 512
num_channels = 1

# define the interation times
generation = 3000

# test the accuracy and loss every 50 times of interation
output_every = 50
eval_every = 50

# define the inital learning rate as 0.002
# define the 
learning_rate = 0.002
lr_decay = 0.9
num_gens_to_wait = 25
#-----------------------#
if __name__ == '__main__':
    # make placeholder for tensorflow to store information for training
    train_input = tf.placeholder(dtype=tf.uint8, shape=[batch_size, 512, 512, num_channels], name='train_input')
    train_target = tf.placeholder(dtype=tf.uint8, shape=[batch_size, 512, 512, num_channels], name='train_target')
    test_input = tf.placeholder(dtype=tf.uint8, shape=[1, 512, 512, num_channels], name='test_input')
    test_target = tf.placeholder(dtype=tf.uint8, shape=[1, 512, 512, num_channels], name='test_input')
    boun_target = tf.placeholder(dtype=tf.uint8, shape=[batch_size, 512, 512, num_channels], name='boun_target')
   
    # load dataset
    path_data = 'train/image'
    path_truth = 'train/gtruth'
    path_loss = 'train/boundary'
    data_set = np.zeros((100, 512, 512))
    target_set = np.zeros((100, 512, 512))
    boundary_set = np.zeros((100, 512, 512))

    for i in range(0, 100):
        # read the information in grayscale images
        data_set[i] = cv2.imread(path_data + '/'+ str(i) + '.bmp', 0)
        target_set[i] = cv2.imread(path_truth + '/'+ str(i) + '.bmp', 0)
        boundary_set[i] = cv2.imread(path_loss + '/'+ str(i) + '.bmp', 0)

    with tf.variable_scope('model_defination') as scope:
        model_output = tf.cast(cnn_model(train_input, batch_size), tf.float32, name = 'train_output')
        scope.reuse_variables()
        test_output = tf.cast(cnn_model(test_input, 1), tf.float32, name = 'test_output')

    loss = loss_function(model_output, train_target, boun_target)
    accuracy = accuracy_of_batch(test_output, test_target)

    generation_num = tf.Variable(0, trainable = False)

    #geneartion_num get the current generation_number
    #learning_rate initial  0.002
    train_op = train_step(loss, generation_num, learning_rate, num_gens_to_wait, lr_decay)
    init = tf.global_variables_initializer()
    sess.run(init)
    tf.train.start_queue_runners(sess = sess)

    train_loss = []
    test_accuracy = []
    for i in range(generation):
        #1-10
        rand_index = np.random.choice((data_set.shape[0]), batch_size+1)
        #input 10 images for training
        rand_train_input = data_set[rand_index[0:batch_size]].reshape((batch_size, 512, 512, 1))
        #10 target images for training
        rand_train_target = target_set[rand_index[0:batch_size]].reshape((batch_size, 512, 512, 1))
        #input 1 iamge for testing
        rand_test_input = data_set[rand_index[-1]].reshape((1, 512, 512, 1))
        #1 target image for testing
        rand_test_target = target_set[rand_index[-1]].reshape((1, 512, 512, 1))
        #10 image for boundary
        rand_boudary_target = boundary_set[rand_index[0:batch_size]].reshape((batch_size, 512, 512, 1))
        # save the cnn model
        saver = tf.train.Saver(max_to_keep=1)
        _, loss_value = sess.run([train_op, loss], feed_dict={train_input:rand_train_input, train_target:rand_train_target, boun_target:rand_boudary_target})
        
        # test the loss value every 50 iterations
        if (i+1) % output_every == 0:
            train_loss.append(loss_value)
            output = 'Generation {}: Loss = {};'.format((i+1), loss_value)
            print(output)

        # test the accuracy rate every 50 iterations
        if (i+1) % eval_every == 0:
            temp_accuracy = sess.run([accuracy], feed_dict={test_input:rand_test_input, test_target:rand_test_target})
            test_accuracy.append(temp_accuracy)
            acc_output = ' --- Test accuracy = {:.2f}%.'.format(temp_accuracy[0])
            print(acc_output)

    saver.save(sess, 'relaynet.ckpt', global_step=generation)

    # plot the accuracy and loss
    x = np.linspace(50, 3000, 60)
    plt.plot(x, test_accuracy)
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.title('Accuracy-step Graph')
    plt.savefig('graph/Accuracy-step Graph.jpg')
    plt.plot(x, train_loss)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss-step Graph')
    plt.savefig('graph/Loss-step Graph.jpg')
