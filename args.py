import argparse

parser = argparse.ArgumentParser(description='iTom')
parser.add_argument('--gpu_id', type=str, nargs='?', default="0")
parser.add_argument('--gpu_frac', type=float, default=0.5,
                    help="fraction of gpu memory to use")
parser.add_argument('--tuning', type=int, default=0,
                    help="`0` for normal training, `1` for fine-tuning/tuning hyper param")
parser.add_argument('--err_only', type=int, default=0,
                    help="whether to log errors only")
parser.add_argument('--n_fold', type=int, default=5)
parser.add_argument('--i_fold', type=int, default=0)
parser.add_argument('--seed', type=int, default=7,
                    help="random seed")
parser.add_argument('--cnnf_weight', type=str,
                    default='/home/dataset/vgg_net.mat',
                    help="CNN-F weights file path")
parser.add_argument('--log_path', type=str, default="./log")
parser.add_argument('--data_path', type=str, default="/home/dataset/nuswide-tc21")
parser.add_argument('--dataset', type=str, default="flickr",
                    help="nuswide_tc21, coco, flickr")

parser.add_argument('--n_class', type=int, default=24)
parser.add_argument('--bit', type=int, default=16)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--pos_thres', type=int, default=-1,
                    help="position threshold (mAP@?, nDCG@?), `-1` means `all`")

parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--iter', type=int, default=60)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--decay_step', type=int, default=10)
parser.add_argument('--decay_rate', type=float, default=0.7)
parser.add_argument('--weight_decay', type=float, default=0.0005)

parser.add_argument('--test_per', type=int, default=500)
parser.add_argument('--save_per', type=int, default=1000)

args = parser.parse_args()
