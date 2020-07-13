import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
import random
import numpy as np
import tensorflow as tf
import flickr
import evaluate
# import losses
from models import *
from wheel import *
from tool import *
from args import *


logger = Logger(args)
if args.err_only:
    tf.logging.set_verbosity(tf.logging.ERROR)
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
os.environ['PYTHONHASHSEED'] = str(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_random_seed(args.seed)
os.system("clear")
print("tf version:", tf.__version__)


dataset = flickr.Flickr()
loop_train = dataset.loop_set("labeled", shuffle=True)
test_labels = dataset.load_label("test")
N_TRAIN = dataset.count("labeled")


gpu_cfg = tf.ConfigProto()
# gpu_cfg.gpu_options.per_process_gpu_memory_fraction = args.gpu_frac
gpu_cfg.gpu_options.allow_growth = True
sess = tf.Session(config=gpu_cfg)


model = Some_Model()
print("trainable:", list(v.name for v in tf.trainable_variables()))


record = Record()
record.add_big("p_pc, r_pc")
record.add_small("l_xent")


def test():
    pass


sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
for epoch in range(args.epoch):
    for _b in range(N_TRAIN // args.batch_size + 1):
        label, image = next(loop_train)
        _, l_xent = sess.run(
            [model.train_op, model.loss_xent],
            feed_dict={model.in_labels: label, model.in_images: image, model.training: True})

    # if n_it % args.test_per == 0:
    logger.log("--- {}: {} ---".format(epoch, timestamp()))
    record.update("l_xent", l_xent)
    test()
    logger.log(record.log_new())


sess.close()
logger.log("--- best ---")
logger.log(record.log_best())
logger.stop()
