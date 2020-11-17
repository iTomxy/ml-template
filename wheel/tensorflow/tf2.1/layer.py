import tensorflow as tf
import tensorflow.keras as K


def conv(x, name, filter_size, stride, in_channel, out_channel, padding='SAME', bias=True, act_fn=tf.nn.relu):
    """conv 2d"""
    with tf.variable_scope(name):
        w = tf.get_variable(
            'weight', shape=[filter_size, filter_size, in_channel, out_channel])
        # w = tf.get_variable('weight_'+name,
        #                     shape=[filter_size, filter_size,
        #                            in_channel, out_channel],
        #                     initializer=tf.initializers.glorot_normal())  # tanh
        #                     #initializer=tf.initializers.he_normal())  # relu
        if bias:
            b = tf.get_variable(
                'bias', initializer=tf.constant(0.1, shape=[num_out]))
            out = tf.nn.xw_plus_b(x, w, b, name=name)
        else:
            out = tf.matmul(x, w, name=name)
        if act_fn:
            out = act_fn(out)

    print(name, ':', out.shape.as_list())
    return out


def fc(units, act_fn=None, bias=True):
    """fully connected layer"""
    # out = L.dense(x, units, act_fn, bias,
    #               kernel_initializer=tf.initializers.truncated_normal(stddev=0.01),
    #             #   kernel_initializer=initializer=tf.initializers.glorot_normal(),  # tanh
    #             #   kernel_initializer=initializer=tf.initializers.he_normal(),  # relu
    #               bias_initializer=tf.initializers.constant(0.1))
    # print(name, ':', out.shape.as_list())
    # return out
    return K.layers.Dense(units, act_fn, bias,
                          kernel_initializer=K.initializers.TruncatedNormal(stddev=0.01),
                          bias_initializer=tf.constant_initializer(0.1))


@tf.custom_gradient
def Htanh(x):
    def grad(dy):
        cond = (-1 <= x) & (x <= 1)
        zero = tf.zeros_like(dy)
        return tf.where(cond, dy, zero)

    return tf.sign(x), grad


@tf.custom_gradient
def pw_threshold(x, epsilon):
    """piece-wise threshold"""
    cond_org = ((0.5 - epsilon) <= x) & (x < (0.5 + epsilon))
    cond_one = x >= (0.5 + epsilon)
    ones = tf.ones_like(x)
    zeros = tf.zeros_like(x)
    y = tf.where(cond_org, x, zeros) + \
        tf.where(cond_one, ones, zeros)

    def grad(dy):
        cond = ((0.5 - epsilon) <= x) & (x < (0.5 + epsilon))
        zeros = tf.zeros_like(dy)
        return tf.where(cond, dy, zeros), epsilon

    return y, grad


class GraphConv(K.layers.Layer):
    """graph convolution layer
    - https://github.com/Megvii-Nanjing/ML-GCN/blob/master/models.py#L7
    - https://zhuanlan.zhihu.com/p/87047648
    """

    def __init__(self, dim_out, act_fn=None, bias=False, k_init=None, b_init=None, name="gc"):
        self.dim_out = dim_out
        # https://github.com/Megvii-Nanjing/ML-GCN/blob/master/models.py#L23
        stdv = 1. / tf.math.sqrt(dim_out)
        if k_init is None:
            # k_init = K.initializers.TruncatedNormal(stddev=0.01)
            k_init = tf.random_uniform_initializer(-stdv, stdv)
        if (b_init is None) and bias:
            # b_init = tf.constant_initializer(0.1)
            b_init = tf.random_uniform_initializer(-stdv, stdv)
        self.fc = K.layers.Dense(dim_out, act_fn, bias, k_init, b_init, name=name)
        super(GraphConv, self).__init__()

    # def build(self, input_shape):
    #     pass

    def call(self, X, A):
        """H = g(AXW [+ b])
        X: [n, d]
        A: [n, n]
        """
        ax = tf.matmul(A, X)
        return self.fc(ax)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.dim_out
        return tf.TensorShape(shape)
