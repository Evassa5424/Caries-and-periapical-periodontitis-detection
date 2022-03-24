import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import tfrecords as jpeg
import numpy as np
import cv2,os,shutil,random

def weight_variable(shape, name):
    initial = tf.random_normal(shape, stddev=0.1) # 变量的初始值为截断正太分布
    return tf.Variable(initial, name=name)

def bias_variable(shape):
    initial = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)
    
def conv_layer(x_image, W_size, weight_name, b_size, bias_name, stride, padding):
    W_conv1 = tf.Variable(tf.random_normal(W_size, stddev=0.1), name=weight_name)
    conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, stride, stride, 1], padding=padding) #54*54
    b_conv1 = tf.Variable(tf.random_normal(b_size, stddev=0.1), name=bias_name)
    h_conv1 = conv1 + b_conv1
    return h_conv1
    
mode = 'train'

bs = 50
num_training_samples = 7124
epoch = round(num_training_samples / bs)
num_batch = epoch * 200


if mode == 'test':
    X_Rays, label, filename_ = jpeg.read_and_decode('test.tfrecords')
    x, y_, filename = tf.train.batch([X_Rays, label, filename_],batch_size=1, capacity=16, num_threads=4)

if mode == 'train':
    X_Rays, label, filename_ = jpeg.read_and_decode('train.tfrecords')
    x, y_, filename = tf.train.shuffle_batch([X_Rays, label, filename_],batch_size=bs, capacity=4096, num_threads=16, min_after_dequeue=512)

if mode == 'validation':
    X_Rays, label, filename_ = jpeg.read_and_decode('validation.tfrecords')
    x, y_, filename = tf.train.batch([X_Rays, label, filename_],batch_size=1, capacity=16, num_threads=4)

y = tf.one_hot(y_, 4)

h_conv1 = conv_layer(x, [5, 5, 1, 8], 'W_conv1', [8], 'b_conv1', 1, 'SAME')
r1 = tf.nn.relu(h_conv1)
pool1 = tf.nn.max_pool(r1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

h_conv2 = conv_layer(pool1, [3, 3, 8, 16], 'W_conv2', [16], 'b_conv2', 1, 'SAME')
r2 = tf.nn.relu(h_conv2)
pool2 = tf.nn.max_pool(r2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

h_conv3 = conv_layer(pool2, [3, 3, 16, 16], 'W_conv3', [16], 'b_conv3', 1, 'SAME')
r3 = tf.nn.relu(h_conv3)
pool3 = tf.nn.max_pool(r3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

pool1_double_pool = tf.nn.max_pool(pool1, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')
concat1 = tf.concat([pool3, pool1_double_pool], axis=3)

h_conv4 = conv_layer(concat1, [3, 3, 16 + 8, 16], 'W_conv4', [16], 'b_conv4', 1, 'SAME')
r4 = tf.nn.relu(h_conv4)
pool4 = tf.nn.max_pool(r4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

h_conv5 = conv_layer(pool4, [3, 3, 16, 16], 'W_conv5', [16], 'b_conv5', 1, 'SAME')
r5 = tf.nn.relu(h_conv5)
pool5 = tf.nn.max_pool(r5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

pool3_double_pool = tf.nn.max_pool(pool3, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')
concat2 = tf.concat([pool5, pool3_double_pool], axis=3)
h_conv6 = conv_layer(concat2, [3, 3, 16 + 16, 8], 'W_conv6', [8], 'b_conv6', 2, 'SAME')

flat = tf.reshape(h_conv6, [-1, 7*7*8])
W_fc = tf.Variable(tf.random_normal([7*7*8, 4], stddev=0.1, dtype=tf.float32), 'W_fc')
b_fc = tf.Variable(tf.random_normal([4], stddev=0.1, dtype=tf.float32), 'b_fc')
fc_add = tf.matmul(flat, W_fc) + b_fc

y_conv = tf.nn.softmax(fc_add)
pred = tf.argmax(y_conv,1)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc_add, labels=y)) # 损失函数，交叉熵
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy, var_list=[var for var in tf.trainable_variables()]) # 使用adam优化

saver=tf.train.Saver(max_to_keep=5000, var_list=[var for var in tf.trainable_variables()])

    
with tf.Session() as sess:
    if mode == 'train':
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        threads = tf.train.start_queue_runners(sess=sess)
        cnt = 1
        l = []
        for i in range(num_batch):
            _, loss = sess.run([train_step, cross_entropy])
            l.append(np.mean(loss))
            if (i+1) % epoch == 0:
                saver.save(sess, 'checkpoints/%d.ckpt'%(cnt))
                print('Save ckpt %d, Loss %g'%(cnt, np.mean(l)))
                cnt += 1
                l = []
    elif mode == 'validation':
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        threads = tf.train.start_queue_runners(sess=sess)
        for cnt in range(1,1+200):
            ACC = 0.
            saver.restore(sess, 'checkpoints/%d.ckpt'%(cnt))
            for i in range(800):
                img_name, predict, gt = sess.run([filename, pred, y_])
                if gt[0] == predict[0]:
                    ACC += 1
            print('\nEpoch = %d'%cnt)
            print('\nValidation Accuracy = %g'%(ACC/800))

    elif mode == 'test':
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        threads = tf.train.start_queue_runners(sess=sess)
        TP_caries = 0.
        FP_caries = 0.
        TN_caries = 0.
        FN_caries = 0.
        TP_periapical_periodontitis = 0.
        FP_periapical_periodontitis = 0.
        TN_periapical_periodontitis = 0.
        FN_periapical_periodontitis = 0.
        checkpoint_num = 153
        saver.restore(sess, 'checkpoints/%d.ckpt'%(checkpoint_num))
        for i in range(800):
            img_name, predict, gt = sess.run([filename, pred, y_])
            if gt[0] == 0:
                if predict[0] == 0:
                    TP_caries += 1
                elif:
                    FN_caries += 1
            elif gt[0] == 1:
                if predict[0] == 1:
                    TN_caries += 1
                elif:
                    FP_caries += 1
            elif gt[0] == 2:
                if predict[0] == 2:
                    TP_periapical_periodontitis += 1
                elif:
                    FN_periapical_periodontitis += 1
            elif gt[0] == 3:
                if predict[0] == 3:
                    TN_periapical_periodontitis += 1
                elif:
                    FP_periapical_periodontitis += 1
        
        # SEN = TP/(TP+FN+1e-5)
        # SPEC = TN/(TN+FP+1e-5)
        # PREC = TP/(TP+FP+1e-5)
        # F1score = 2*PREC*SEN/(PREC+SEN+1e-5)
        print('Test performance:\n')
        print('Caries:\n')
        print('TP = %g, FN = %g, TN = %g, FP = %g\n'%(TP_caries,FN_caries,TN_caries,FP_caries))
        print('Periapical periodontitis:\n')
        print('TP = %g, FN = %g, TN = %g, FP = %g\n'%(TP_periapical_periodontitis,FN_periapical_periodontitis,TN_periapical_periodontitis,FP_periapical_periodontitis))
        
