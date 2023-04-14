import collections
import tensorflow as tf
import tensorflow.keras as K


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
