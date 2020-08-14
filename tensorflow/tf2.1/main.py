import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
import random
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
# import cnnf
import voc2007
import evaluate
import losses
from mlgcn import *
from wheel import *
from tool import *
from args import *


logger = Logger(args)
logger.log("tf version: {}".format(tf.__version__))
if args.err_only:
    tf.get_logger().setLevel(logging.ERROR)
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
# gpu_config = [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)]
gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for _gpu in gpus:
    tf.config.experimental.set_memory_growth(_gpu, True)
    # tf.config.experimental.set_virtual_device_configuration(_gpu, gpu_config)
os.environ['PYTHONHASHSEED'] = str(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)
os.system("clear")


dataset = voc2007.VOC2007(zero_as=1)
loop_train = dataset.loop_set("train", shuffle=True)
test_labels = dataset.load_label("test")
labels_w2v = dataset.load_label_w2v()  # [20, 300]
N_TRAIN = dataset.count("train")


model = GCN_ResNet(get_A(args.adj_file, 0.4))
model.build(input_shape=(None, 448, 448, 3))
# model.build(input_shape=(None, 2048))
# model.summary()
# print("trainable:", [v.name for v in model.trainable_variables])

avg_xent = K.metrics.Mean(name='xent_loss')

# optimizer = K.optimizers.Adam()
var_list_w = [v for v in model.trainable_variables if 'kernel' in v.name]
var_base, var_gc = [], []
for v in model.trainable_variables:
    if "graph_conv" in v.name:
        var_gc.append(v)
    else:
        var_base.append(v)
parameters = var_base + var_gc
n_var_base = len(var_base)
optim_base = K.optimizers.SGD(learning_rate=LR_MLGCN(args.lr * args.lrp,
    decay_steps=args.decay_step * (N_TRAIN // args.batch_size),
    decay_rate=args.decay_rate), momentum=args.momentum)
optim_gc = K.optimizers.SGD(learning_rate=LR_MLGCN(args.lr,
    decay_steps=args.decay_step * (N_TRAIN // args.batch_size),
    decay_rate=args.decay_rate), momentum=args.momentum)

record = Record()
record.add_small("ham_loss", "coverage", "rank_loss")
record.add_big("map", "op", "or", "of1", "cp", "cr", "cf1", "acc")


@tf.function
def train(label, image):
    with tf.GradientTape() as tape:
        logit = model(image, labels_w2v)
        loss_xent = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=label, logits=logit))
        loss_reg = args.weight_decay * \
            tf.reduce_mean([tf.nn.l2_loss(v) for v in var_list_w])
        loss = loss_xent + loss_reg

    gradients = tape.gradient(loss, parameters)
    optim_base.apply_gradients(zip(gradients[:n_var_base], var_base))
    optim_gc.apply_gradients(zip(gradients[n_var_base:], var_gc))
    avg_xent(loss_xent)

    return loss_xent


def test():
    Y = test_labels
    score_list = []
    for label, image in dataset.iter_set("test"):
        logit = model(image)
        logit = tf.math.sigmoid(logit)
        score_list.append(logit.numpy())
    score = np.vstack(score_list)  # [n, c]
    pred = (score > 0.5).astype(np.float32)

    ham_loss = skm.hamming_loss(Y, pred)
    coverage = skm.coverage_error(Y, score)
    rank_loss = skm.label_ranking_loss(Y, score)
    # lrap = skm.label_ranking_average_precision_score(Y, score)
    # micro_f1 = skm.f1_score(Y, pred, average="micro")
    # macro_f1 = skm.f1_score(Y, pred, average="macro")
    ap = evaluate.ap_pc(Y, score)
    logger.log("AP: {}".format(ap))
    mAP = ap.mean()
    OP, OR, OF1, CP, CR, CF1, acc, acc_pc = evaluate.prfa(Y, pred)
    logger.log("acc: {}".format(acc_pc))
    # acc = skm.accuracy_score(cast_single(to_matrix(Y)), cast_single(to_matrix(pred)))
    acc_pc, true_pc, pred_pc = evaluate.analyse_outcast(
        cast_single(to_matrix(Y)), cast_single(logit_3d))
    print("acc_pc:", acc_pc)
    print("true_pc:", true_pc)
    print("pred_pc:", pred_pc)

    record.update("ham_loss", ham_loss)
    record.update("coverage", coverage)
    record.update("rank_loss", rank_loss)
    record.update("map", mAP)
    record.update("op", OP)
    record.update("or", OR)
    record.update("of1", OF1)
    record.update("cp", CP)
    record.update("cr", CR)
    record.update("cf1", CF1)
    record.update("acc", acc)
    logger.log(record.log_new())


for epoch in range(args.epoch):
    while _b in range(N_TRAIN // args.batch_size + 1):
        label, image = next(loop_train)
        l_xent = train(label, image)

    # if n_it % args.test_per == 0:
    logger.log("--- {}: {} ---".format(epoch, timestamp()))
    logger.log("xent: {}".format(l_xent.numpy()))
    logger.log("avg xent: {}".format(avg_xent.result()))
    test()


logger.log("--- best ---")
logger.log(record.log_best())
logger.stop()
