import tensorflow as tf


def conv(x, name, filter_size, stride, in_channel, out_channel, padding='SAME', act_fn=tf.nn.relu):
    """conv 2d"""
    with tf.variable_scope(name):
        w = tf.get_variable(
            'kernel_'+name, shape=[filter_size, filter_size, in_channel, out_channel])
        # w = tf.get_variable('weight_'+name,
        #                     shape=[filter_size, filter_size,
        #                            in_channel, out_channel],
        #                     initializer=tf.initializers.glorot_normal())  # tanh
        #                     #initializer=tf.initializers.he_normal())  # relu
        b = tf.get_variable('bias_'+name, shape=[out_channel])

        out = tf.nn.conv2d(x, w,
                           strides=[1, stride, stride, 1],
                           padding=padding)
        out = tf.reshape(tf.nn.bias_add(out, b),
                         [-1]+out.get_shape().as_list()[1:])
        if act_fn:
            out = act_fn(out)

    print(name, ':', out.shape.as_list())
    return out


def fc(x, name, num_in, num_out, act_fn=None, bias=True, stddev=0.01):
    """fully connected layer"""
    with tf.variable_scope(name):
        w = tf.get_variable('kernel_'+name,
                            #shape=[num_in, num_out],
                            #initializer=tf.initializers.glorot_normal())  # tanh
                            # initializer=tf.initializers.he_normal())  # relu
                            initializer=tf.truncated_normal([num_in, num_out], stddev=stddev))
        out = tf.matmul(x, w)
        if bias:
            b = tf.get_variable(
                'bias_'+name, initializer=tf.constant(0.1, shape=[num_out]))
            out = out + b
        if act_fn:
            out = act_fn(out)

    print(name, ':', out.shape.as_list())
    return out


def gcn(DAD, x, num_in, num_out,
        act_fn=None, bias=False, k_init=None, b_init=None, name="GCN"):
    """GCN layer
    DAD: $D^{-1/2} A D^{-1/2}$
    """
    _std = 1. / math.sqrt(num_out)
    if k_init is None:
        k_init = tf.initializers.random_uniform(- _std, _std)
    if bias and (b_init is None):
        b_init = tf.initializers.random_uniform(- _std, _std)
    h = tf.matmul(DAD, x)
    return tf.layers.dense(h, num_out, act_fn, bias, k_init, b_init, name=name)


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
        S = tf.cast(tf.equal(label[:, None], label2[None, :]), "float32")
    else:
        S = tf.cast(tf.matmul(label, tf.transpose(label2)) > 0, "float32")
    return S


def cos(X, Y=None):
    """cosine of every (Xi, Yj) pair
    X, Y: (n, dim)
    """
    # X_n = X / np.linalg.norm(X, ord=2, axis=1)[:, np.newaxis]
    # Y_n = Y / np.linalg.norm(Y, ord=2, axis=1)[:, np.newaxis]
    # return np.dot(X_n, np.transpose(Y_n))
    X_n = tf.math.l2_normalize(X, axis=1)
    if (Y is None) or (X is Y):
        return tf.matmul(X_n, tf.transpose(X_n))
    Y_n = tf.math.l2_normalize(Y, axis=1)
    _cos = tf.matmul(X_n, tf.transpose(Y_n))
    return tf.clip_by_value(_cos, -1, 1)


def hamming(X, Y=None, discrete=False):
    if Y is None:
        Y = X
    K = tf.cast(tf.shape(X)[1], "float32")
    kernel = K - tf.matmul(X, tf.transpose(Y))
    if discrete:
        H = tf.cast(kernel, "int32") // 2
        return tf.cast(H, "float32")
    else:
        return 0.5 * kernel


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
