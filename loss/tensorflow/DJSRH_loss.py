import tensorflow as tf
from args import args


def DJSRH_loss(fc7, fc7_u, hc, hc_u):
    fc7_ul = tf.concat([fc7_u, fc7], axis=0)
    S = cos(fc7_ul, fc7_ul)  # * 2 - 1
    S = (1 - args.eta) * S + args.eta * \
        tf.matmul(S, tf.transpose(S)) / (args.batch_size + args.batch_size_u)

    hc_ul = tf.concat([hc, hc_u], axis=0)
    BtB = cos(hc_ul, hc_ul)

    return tf.losses.mean_squared_error(args.mu * S, BtB)
