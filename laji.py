import tensorflow as tf

import numpy as np
import os
import cv2
import random

img_width ,img_height = 224,224
batch_size = 40
baches = 21
weight_decay = 0.5
is_train=True
reuse=False

def conv2d(input_, output_dim, k_h, k_w, d_h, d_w, stddev=0.02, name='conv2d', bias=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],regularizer=tf.contrib.layers.l2_regularizer(0.5),initializer=tf.truncated_normal_initializer(stddev=stddev))
        #truncated_normal_initializer生成截断正态分布的随机数
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        if bias:
            biases = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)

        return conv

def relu(x, name='relu6'):
    return tf.nn.relu6(x, name)


def batch_norm(x, momentum=0.9, epsilon=1e-5, train=True, name='bn'):
    return tf.layers.batch_normalization(x,
                      momentum=momentum,
                      epsilon=epsilon,
                      scale=True,
                      training=train,
                      name=name)

def res_block(input, expansion_ratio, output_dim, stride, is_train, name, bias=False, shortcut=True):
    with tf.name_scope(name), tf.variable_scope(name):
        # pw
        bottleneck_dim=round(expansion_ratio*input.get_shape().as_list()[-1])
        net = conv_1x1(input, bottleneck_dim, name='pw', bias=bias)
        net = batch_norm(net, train=is_train, name='pw_bn')
        net = relu(net)
        # dw
        net = dwise_conv(net, strides=[1, stride, stride, 1], name='dw', bias=bias)
        net = batch_norm(net, train=is_train, name='dw_bn')
        net = relu(net)
        # pw & linear
        net = conv_1x1(net, output_dim, name='pw_linear', bias=bias)
        net = batch_norm(net, train=is_train, name='pw_linear_bn')

        # element wise add, only for stride==1
        if shortcut and stride == 1:
            in_dim=int(input.get_shape().as_list()[-1])
            if in_dim != output_dim:
                ins=conv_1x1(input, output_dim, name='ex_dim')
                net=ins+net
            else:
                net=input+net

        return net

def dwise_conv(input, k_h=3, k_w=3, channel_multiplier= 1, strides=[1,1,1,1],
               padding='SAME', stddev=0.02, name='dwise_conv', bias=False):
    with tf.variable_scope(name):
        in_channel=input.get_shape().as_list()[-1]
        w = tf.get_variable('w', [k_h, k_w, in_channel, channel_multiplier],
                        regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.depthwise_conv2d(input, w, strides, padding, rate=None,name=None,data_format=None)
        if bias:
            biases = tf.get_variable('bias', [in_channel*channel_multiplier], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)

        return conv

def conv_1x1(input, output_dim, name, bias=False):
    with tf.name_scope(name):
        return conv2d(input, output_dim, 1,1,1,1, stddev=0.02, name=name, bias=bias)

def global_avg(x):
    with tf.name_scope('global_avg'):
        net=tf.layers.average_pooling2d(x, x.get_shape()[1:-1], 1)
        return net

def conv2d_block(input, out_dim, k, s, is_train, name):
    with tf.name_scope(name), tf.variable_scope(name):
        net = conv2d(input, out_dim, k, k, s, s, name='conv2d')
        net = batch_norm(net, train=is_train, name='bn')
        net = relu(net)
        return net

def mobilenetv2(inputs, num_classes):  #input 224*224*3
    exp = 6  # expansion ratio
    with tf.variable_scope('mobilenetv2'):
        net = conv2d_block(inputs, 32, 3, 2, is_train, name='conv1_1')  # size/2

        net = res_block(net, 1, 16, 1, is_train, name='res2_1')

        net = res_block(net, exp, 24, 2, is_train, name='res3_1')  # size/4
        net = res_block(net, exp, 24, 1, is_train, name='res3_2')

        net = res_block(net, exp, 32, 2, is_train, name='res4_1')  # size/8
        net = res_block(net, exp, 32, 1, is_train, name='res4_2')
        net = res_block(net, exp, 32, 1, is_train, name='res4_3')

        net = res_block(net, exp, 64, 1, is_train, name='res5_1')
        net = res_block(net, exp, 64, 1, is_train, name='res5_2')
        net = res_block(net, exp, 64, 1, is_train, name='res5_3')
        net = res_block(net, exp, 64, 1, is_train, name='res5_4')

        net = res_block(net, exp, 96, 2, is_train, name='res6_1')  # size/16
        net = res_block(net, exp, 96, 1, is_train, name='res6_2')
        net = res_block(net, exp, 96, 1, is_train, name='res6_3')

        net = res_block(net, exp, 160, 2, is_train, name='res7_1')  # size/32
        net = res_block(net, exp, 160, 1, is_train, name='res7_2')
        net = res_block(net, exp, 160, 1, is_train, name='res7_3')

        net = res_block(net, exp, 320, 1, is_train, name='res8_1', shortcut=False)

        net = pwise_block(net, 1280, is_train, name='conv9_1')
        net = global_avg(net)
        logits = tf.contrib.layers.flatten(conv_1x1(net, num_classes, name='logits'))

        pred = tf.nn.softmax(logits, name='prob')
        return logits, pred

def pwise_block(inputs, filters, is_train, stride = 1, name=''):
    net = conv_1x1(inputs, filters, name='pw', bias=False)
    net = batch_norm(net, train=is_train, name='pw_bn')            
    return net
def read_data(dpath,nlist):
    dlist = []
    for row in nlist:
        img = cv2.imread(dpath+row)
        dlist.append(img)
    return np.array(dlist)
def main():
    data_path = 'after/'
	#laji = os.listdir(data_path)
    all_obj = []
    trains=[]
    labels=[]
    z = 0
    #for i in laji:
    all_pic = os.listdir(data_path)
    for pic in all_pic :
        image = cv2.imread(data_path+pic)
        trains.append(pic)
        if pic[0:-5] not in all_obj :
            all_obj.append(pic[0:-5])
            z +=1
            labels.append(z)
        else :
            labels.append(all_obj.index(pic[0:-5]))
    labels = [np.eye(z)[a-1] for a in labels ]
    trains = np.array(trains)
    labels = np.array(labels)
    inputs = tf.placeholder(tf.float32,[None,224,224,3],name = 'inputs')
    print(inputs.shape)
    logits, pred = mobilenetv2(inputs,num_classes=len(all_obj))
    sess = tf.Session()
    ys = tf.placeholder(tf.int32, [None,z])
    

    loss = tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels = ys)
    loss = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)
    pred_y = tf.argmax(pred, 1,name='output')
    correct = tf.equal(pred_y, tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    saver = tf.train.Saver(max_to_keep = 4, keep_checkpoint_every_n_hours = 1)
    tf.train.write_graph(sess.graph_def, "model/", "graph.pb", as_text=False)
    sess.run(tf.global_variables_initializer())
    n = len(trains)
    for i in range(0,baches) :
        random.seed(5)
        random.shuffle(trains)
        random.shuffle(labels)
        for j in range(0,int(n/batch_size)):
            x = trains[batch_size*j:batch_size*(j+1)]
            y = labels[batch_size*j:batch_size*(j+1)]
            if len(trains)-batch_size*j<batch_size:
                x = trains[batch_size*j:-1]
                y = labels[batch_size*j:-1]
            x = read_data(data_path,x)
            _,losses,acc= sess.run([optimizer,loss,accuracy],feed_dict={inputs : x,ys:y})
            print('epoch:',i,'step:',j,'loss:',losses,'acc:',acc)
        if i %5==0:
            saver.save(sess, 'model/model.ckpt', write_meta_graph = True)


if __name__=='__main__':
    main()
