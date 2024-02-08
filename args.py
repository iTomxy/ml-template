import argparse

"""argparse examples"""

parser = argparse.ArgumentParser(description='iTom')
parser.add_argument('--gpu', type=str, nargs='?', default="0", help='to set os.environ["CUDA_VISIBLE_DEVICES"]')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--tune', action="store_true", help="enable tuning mode")
parser.add_argument('--log_path', type=str, default="./log")
parser.add_argument('--dataset', type=str, default="flickr25k", choices=["flickr25k", "nuswide_tc21"])
parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
parser.add_argument('--decay_step', type=int, nargs='+', default=[50, 100], help="lr decay steps")
args = parser.parse_args()
