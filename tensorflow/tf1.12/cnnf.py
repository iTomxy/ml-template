import tensorflow as tf
import scipy.misc
import numpy as np
import scipy.io


def CNN_F(input_image, weight_file, training, top=True, var_scope="CNN_F", reuse=False):
    data = scipy.io.loadmat(weight_file)['net'][0][0]
    layers = ('conv1', 'relu1', 'norm1', 'pool1',
              'conv2', 'relu2', 'norm2', 'pool2',
              'conv3', 'relu3',
              'conv4', 'relu4',
              'conv5', 'relu5', 'pool5',
              'fc6', 'relu6',
              'fc7', 'relu7')
    weights = data[0][0]
    net = {}
    ops = []
    current = tf.convert_to_tensor(input_image, dtype='float32')
    with tf.variable_scope(var_scope, reuse=reuse):
        for i, name in enumerate(layers):
            if name.startswith('conv'):
                # assert weights[i][0][0][0].startswith('conv')
                kernels, bias = weights[i][0][0][2][0]
                # matconvnet: weights are [width, height, in_channels, out_channels]
                # tensorflow: weights are [height, width, in_channels, out_channels]
                #kernels = np.transpose(kernels, (1, 0, 2, 3))

                bias = bias.reshape(-1)
                pad = weights[i][0][0][4]
                stride = weights[i][0][0][5]
                current = _conv_layer(current, kernels, bias,
                                      pad, stride, i, ops, net)
            elif name.startswith('relu'):
                # assert weights[i][0][0][0].startswith('relu')
                current = tf.nn.relu(current)
                if name == "relu6" or name == "relu7":
                    # current = tf.nn.dropout(current, keep_prob=keep_prob, name="dropout"+name[-1])
                    current = tf.layers.dropout(
                        current, rate=0.5, training=training, name="dropout"+name[-1])
            elif name.startswith('pool'):
                # assert weights[i][0][0][0].startswith('pool')
                stride = weights[i][0][0][4]
                pad = weights[i][0][0][5]
                area = weights[i][0][0][3]
                current = _pool_layer(current, stride, pad, area)
            elif name.startswith('fc'):
                if top:
                    # assert weights[i][0][0][0].startswith('fc')
                    kernels, bias = weights[i][0][0][2][0]
                    # matconvnet: weights are [width, height, in_channels, out_channels]
                    # tensorflow: weights are [height, width, in_channels, out_channels]
                    #kernels = np.transpose(kernels, (1, 0, 2, 3))

                    bias = bias.reshape(-1)
                    # with tf.name_scope(name):
                    current = _full_conv(current, kernels, bias, i, ops, net)
            elif name.startswith('norm'):
                # assert weights[i][0][0][0].startswith('norm')
                current = tf.nn.local_response_normalization(
                    current, depth_radius=2, bias=2.000, alpha=0.0001, beta=0.75)
            net[name] = current

        fc7 = current
        if top:
            # fc7 = tf.squeeze(fc7)
            fc7 = tf.reshape(fc7, [-1, fc7.shape.as_list()[-1]], name="fc7")  # [None, 4096]
        # else:
        #     fc7 = tf.layers.flatten(fc7, name="pool5")  # [None, 6 * 6 * 256]

    print("CNN-F output:", fc7.shape.as_list())
    return fc7


"""暂时用不到
def CNN_F(input_image, weight_file, keep_prob, var_scope="CNN_F", reuse=False):
    data = scipy.io.loadmat(weight_file)
    layers = ('conv1', 'relu1', 'norm1', 'pool1',
              'conv2', 'relu2', 'norm2', 'pool2',
              'conv3', 'relu3',
              'conv4', 'relu4',
              'conv5', 'relu5', 'pool5',
              'fc6', 'relu6',
              'fc7', 'relu7')  # , 'fc8')
    weights = data['layers'][0]
    # mean = data['normalization'][0][0][0]
    net = {}
    ops = []
    current = tf.convert_to_tensor(input_image, dtype='float32')
    with tf.variable_scope(var_scope, reuse=reuse):
        for i, name in enumerate(layers):
            if name.startswith('conv'):
                kernels, bias = weights[i][0][0][0][0]
                # matconvnet: weights are [width, height, in_channels, out_channels]
                # tensorflow: weights are [height, width, in_channels, out_channels]
                #kernels = np.transpose(kernels, (1, 0, 2, 3))

                bias = bias.reshape(-1)
                pad = weights[i][0][0][1]
                stride = weights[i][0][0][4]
                current = _conv_layer(current, kernels, bias,
                                      pad, stride, i, ops, net)
            elif name.startswith('relu'):
                current = tf.nn.relu(current)
                if name == "relu6" or name == "relu7":
                    # current = tf.layers.dropout(
                    #     current, rate=0.5,
                    #     training=is_training,
                    #     name="dropout"+name[-1])
                    current = tf.nn.dropout(current,
                                            keep_prob=keep_prob,
                                            name="dropout"+name[-1])
            elif name.startswith('pool'):
                stride = weights[i][0][0][1]
                pad = weights[i][0][0][2]
                area = weights[i][0][0][5]
                current = _pool_layer(current, stride, pad, area)
            elif name.startswith('fc'):
                kernels, bias = weights[i][0][0][0][0]
                # matconvnet: weights are [width, height, in_channels, out_channels]
                # tensorflow: weights are [height, width, in_channels, out_channels]
                #kernels = np.transpose(kernels, (1, 0, 2, 3))

                bias = bias.reshape(-1)
                # with tf.name_scope(name):
                current = _full_conv(current, kernels, bias, i, ops, net)
            elif name.startswith('norm'):
                current = tf.nn.local_response_normalization(
                    current, depth_radius=2, bias=2.000, alpha=0.0001, beta=0.75)
            net[name] = current

        fc7 = current
        # fc7 = tf.squeeze(fc7)
        fc7 = tf.reshape(fc7, [-1, fc7.shape.as_list()[-1]], name="cnnf_output")

    return fc7
"""


def _conv_layer(input, weights, bias, pad, stride, i, ops, net):
    pad = pad[0]
    stride = stride[0]
    input = tf.pad(input, [[0, 0], [pad[0], pad[1]], [
                   pad[2], pad[3]], [0, 0]], "CONSTANT")
    # w = tf.Variable(weights, name='w'+str(i), dtype='float32')
    # b = tf.Variable(bias, name='bias'+str(i), dtype='float32')
    w = tf.get_variable('kernel'+str(i), shape=weights.shape, dtype='float32',
                        initializer=tf.initializers.constant(weights))
    b = tf.get_variable('bias'+str(i), shape=bias.shape, dtype='float32',
                        initializer=tf.initializers.constant(bias))
    ops.append(w)
    ops.append(b)
    net['weights' + str(i)] = w
    net['b' + str(i)] = b
    conv = tf.nn.conv2d(input, w, strides=[
                        1, stride[0], stride[1], 1], padding='VALID', name='conv'+str(i))
    return tf.nn.bias_add(conv, b, name='add'+str(i))


def _full_conv(input, weights, bias, i, ops, net):
    # w = tf.Variable(weights, name='w' + str(i), dtype='float32')
    # b = tf.Variable(bias, name='bias' + str(i), dtype='float32')
    w = tf.get_variable('kernel'+str(i), shape=weights.shape, dtype='float32',
                        initializer=tf.initializers.constant(weights))
    b = tf.get_variable('bias'+str(i), shape=bias.shape, dtype='float32',
                        initializer=tf.initializers.constant(bias))
    ops.append(w)
    ops.append(b)
    net['weights' + str(i)] = w
    net['b' + str(i)] = b
    conv = tf.nn.conv2d(input, w, strides=[
                        1, 1, 1, 1], padding='VALID', name='fc'+str(i))
    return tf.nn.bias_add(conv, b, name='add'+str(i))


def _pool_layer(input, stride, pad, area):
    pad = pad[0]
    area = area[0]
    stride = stride[0]
    input = tf.pad(input, [[0, 0], [pad[0], pad[1]], [
                   pad[2], pad[3]], [0, 0]], "CONSTANT")
    return tf.nn.max_pool(input, ksize=[1, area[0], area[1], 1], strides=[1, stride[0], stride[1], 1], padding='VALID')


def preprocess(image, mean_pixel):
    return image - mean_pixel


def unprocess(image, mean_pixel):
    return image + mean_pixel


def get_meanpix(data_path):
    data = scipy.io.loadmat(data_path)
    mean = data['normalization'][0][0][0]
    return mean
