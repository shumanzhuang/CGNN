import argparse
from utils import  str2bool

def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",type=str,default="0",help="Device:cuda:num or cpu")
    parser.add_argument("--dataset",type=str,default='ACM3025_0', help="name of datasets")
    parser.add_argument('--normalize_features', type=str2bool, default=True)
    parser.add_argument('--ptb_rate', type=float, default=0.25, help="Perturbation rate")
    parser.add_argument("--n_repeated", type=int, default=1, help="Number of repeated times.")
    parser.add_argument('--num_hidden', type=int ,default=32)
    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout rate.")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--weight_decay', type=int, default=5e-4)
    parser.add_argument('--layer_num', type=int ,default=2)
    parser.add_argument('--max_epoch', type=int, default=800)
    parser.add_argument('--patience',type=int,default=200)
    parser.add_argument("--path", type=str, default="./datasets/multi_relational/",
                        help="Path of datasets")
    parser.add_argument("--knns", type=int, default=10, help="Number of k nearest neighbors")
    parser.add_argument("--pr1", action='store_true', default=True, help="Using prunning strategy 1 or not")
    parser.add_argument("--pr2", action='store_true', default=True, help="Using prunning strategy 2 or not")
    parser.add_argument("--common_neighbors", type=int, default=2,
                        help="Number of common neighbors")
    parser.add_argument("--threshold_c", type=float, default=0.01, help="threshold for model para, default: 0.01 [use all]")
    parser.add_argument("--threshold", type=float, default=1 / 28, help="threshold for model para, default: 1/28 [use all]")
    parser.add_argument('--seed', type=int, default=16)
    parser.add_argument("--shuffle_seed", type=int, default=16, help="Random seed for train-test split.")
    parser.add_argument("--ratio", type=float, default=0.2)
    args = parser.parse_args()
    return args