import collections
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


def euclidean(A, B=None, sqrt=False):
    if (B is None) or (B is A):
        aTb = tf.matmul(A, tf.transpose(A))
        aTa = bTb = tf.linalg.diag_part(aTb)
    else:
        aTb = tf.matmul(A, tf.transpose(B))
        aTa = tf.linalg.diag_part(tf.matmul(A, tf.transpose(A)))
        bTb = tf.linalg.diag_part(tf.matmul(B, tf.transpose(B)))

    D = aTa[:, None] - 2.0 * aTb + bTb[None, :]
    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    D = tf.maximum(D, 0.0)

    if sqrt:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.cast(tf.equal(D, 0.0), "float32")
        D = D + mask * 1e-16
        D = tf.math.sqrt(D)
        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        D = D * (1.0 - mask)

    return D


def rbf_kernel(X, Y, sigma=1.25):
    """K(i,j) = exp(-0.5 * ||xi - xj||^2 / sigma^2)
    sigma: width of RBF kernel, default 1.25 from paper:
        - Learning with Local and Global Consistency
    """
    D = euclidean_dist(X, Y)
    return tf.math.exp(-0.5 * D / sigma**2)


def sim_mat(label, label2=None, sparse=False):
    """similarity matrix
    S[i][j] = 1 <=> i- & j-th share at lease 1 label
    S[i][j] = 0 otherwise
    """
    label2 = label if (label2 is None) else label2
    if sparse:
        S = tf.cast(label[:, None] == label2[None, :], "float32")
    else:
        S = tf.cast(tf.matmul(label, tf.transpose(label2)) > 0, "float32")
    return S


def cos(X, Y=None):
    """cosine of every (Xi, Yj) pair
    X, Y: (n, dim)
    """
    X_n = tf.math.l2_normalize(X, axis=1)
    if (Y is None) or (X is Y):
        return tf.matmul(X_n, tf.transpose(X_n))
    Y_n = tf.math.l2_normalize(Y, axis=1)
    _cos = tf.matmul(X_n, tf.transpose(Y_n))
    return tf.clip_by_value(_cos, -1, 1)


def hamming(X, Y=None):
    if Y is None:
        Y = X
    K = tf.cast(tf.shape(X)[1], "float32")
    return (K - tf.matmul(X, tf.transpose(Y))) / 2


def top_k_mask(D, k, rand_pick=False):
    """M[i][j] = 1 <=> D[i][j] is oen of the BIGGEST k in i-th row
    Args:
        D: (n, n), distance matrix
        k: param `k` of kNN
        rand_pick: true or false
            - if `True`, only ONE of the top-K element in each row will be selected randomly;
            - if `False`, ALL the top-K elements will be selected as usual.
    Ref:
        - https://cloud.tencent.com/developer/ask/196899
        - https://blog.csdn.net/HackerTom/article/details/103587415
    """
    n_row = tf.shape(D)[0]
    n_col = tf.shape(D)[1]

    k_val, k_idx = tf.math.top_k(D, k)
    if rand_pick:
        c_idx = tf.random_uniform([n_row, 1],
                                  minval=0, maxval=k,
                                  dtype="int32")
        r_idx = tf.range(n_row, dtype="int32")[:, None]
        idx = tf.concat([r_idx, c_idx], axis=1)
        k_idx = tf.gather_nd(k_idx, idx)[:, None]

    idx_offset = (tf.range(n_row) * n_col)[:, None]
    k_idx_linear = k_idx + idx_offset
    k_idx_flat = tf.reshape(k_idx_linear, [-1, 1])

    updates = tf.ones_like(k_idx_flat[:, 0], "int32")
    mask = tf.scatter_nd(k_idx_flat, updates, [n_row * n_col])
    mask = tf.reshape(mask, [-1, n_col])
    # mask = tf.cast(mask, "bool")

    return mask


def check_nan_inf(tensors):
    """status = {0: OK, 1: NaN, 2: inf, 3: NaN & inf}
    tensors: single tensor, or collections of tensor
    return: {1: nan, 2: inf}
    """
    if isinstance(tensors, tf.Tensor):
        _nan = tf.reduce_any(tf.is_nan(tensors))
        _inf = tf.reduce_any(tf.is_nan(tensors))
        res = tf.reduce_sum(tf.where(
            tf.stack([_nan, _inf]), tf.constant([1, 2]), tf.constant([0, 0])))
        if res:
            return res
    elif isinstance(tensors, collections.Iterable):
        for _t in tensors:
            res = check_nan_inf(_t)
            if res:
                return res
    else:
        raise Exception("check_nan_inf: unsupported type: {}".format(type(tensors)))
    return 0
