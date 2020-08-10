import os
import random
import numpy as np
import torch
import flickr
import evaluate
import pre_process as prep
import losses
from csq import *
from wheel import *
from tool import *
from args import *


print(torch.__version__)               # PyTorch version
print(torch.version.cuda)              # Corresponding CUDA version
print(torch.backends.cudnn.version())  # Corresponding cuDNN version
print(torch.cuda.get_device_name(0))   # GPU type


logger = Logger(args)
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
os.environ['PYTHONHASHSEED'] = str(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
os.system("clear")
print("pytorch version:", torch.__version__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = flickr.Flickr()
# loop_train = dataset.loop_set("labeled", shuffle=True)
C = dataset.load_centre().cuda()  # [c, #bit]
C_random = torch.randint_like(C[0], 2) * 2 - 1  # {-1, 1}
N_TRAIN = dataset.count("labeled")

train_transform = prep.image_train(resize_size=255, crop_size=224)
test_transform = prep.image_test(resize_size=255, crop_size=224)


if args.model_type == 'resnet50' or args.model_type == 'resnet152':
    model = Model().cuda()
elif args.model_type == 'alexnet':
    model = AlexNetFc().cuda()
# model = model.to(device)
# model = torch.nn.DataParallel(model).cuda()


bce_loss = nn.BCELoss().cuda()
#criterion = nn.MSELoss().cuda()
params_list = [{'params': model.feature_layers.parameters(), 'lr': args.lrp * args.lr},  # 0.05*(args.lr)
               {'params': model.hash_layer.parameters()}]
optimizer = torch.optim.Adam(params_list, lr=args.lr, betas=(0.9, 0.999))


record = Record()
record.add_big("map_ham")
record.add_small("l_xent", "l_quant", "l_pair")


def gen_hash(loader):
    model.eval()
    h_ls, y_ls = [], []
    with torch.no_grad():
        for label, image in loader:
            y_ls.append(label)
            h = model(image.cuda())
            h[h >= args.threshold] = 1
            h[h < args.threshold] = -1
            h_ls.append(h.cpu().numpy())
    return np.vstack(y_ls), np.vstack(h_ls)


def test():
    qL, qH = gen_hash(dataset.iter_set("test", test_transform))
    rL, rH = gen_hash(dataset.iter_set("ret", test_transform))
    mAP_ham = evaluate.calc_mAP(qH, rH, qL, rL, 1, args.mAP_at)
    record.update("map_ham", mAP_ham)


for epoch in range(args.epoch):
    adjust_learning_rate(optimizer, epoch)
    model.train()
    for label, image in dataset.iter_set("labeled", train_transform):
        label, image = label.cuda(), image.cuda()
        centre = multi_centre(C, label, C_random)  # [n, bit], {-1, 1}
        optimizer.zero_grad()
        hc = model(image)

        _C = 0.5 * (centre + 1)
        _H = 0.5 * (hc + 1)
        loss_centre = bce_loss(_H, _C)
        loss_quant = ((hc.abs() - 1.0) ** 2).mean()
        _n = label.size(0)
        _X1, _X2 = hc[:_n], hc[_n:]
        _Y1, _Y2 = label[:_n], label[_n:]
        loss_similarity = losses.struct_loss(
            _X1, _X2, sim_mat(_Y1, _Y2), 10. / args.bit)
        loss = args.lambda0 * loss_centre + \
            args.lambda1 * loss_similarity + \
            args.lambda2 * loss_quant
        loss.backward()
        optimizer.step()

    logger.log("--- {}: {} ---".format(epoch, timestamp()))
    logger.log("loss: {}".format(loss.cpu().item()))
    logger.log("l_xent: {}".format(loss_centre.cpu().item()))
    logger.log("l_quant: {}".format(loss_quant.cpu().item()))
    logger.log("loss_pair: {}".format(loss_similarity.cpu().item()))
    if epoch % args.test_per == 0:
        record.update("l_xent", loss_centre.cpu().item())
        record.update("l_quant", loss_quant.cpu().item())
        record.update("l_pair", loss_similarity.cpu().item())
        test()
        logger.log(record.log_new())


logger.log("--- best ---")
logger.log(record.log_best())
torch.cuda.empty_cache()
