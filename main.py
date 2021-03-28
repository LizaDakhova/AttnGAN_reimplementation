import argparse
import numpy as np
import os
import random
import torch
import train_DAMSM
import run_AttnGAN

from config import cfg
import torch.backends.cudnn as cudnn
from warnings import filterwarnings

filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Try AttnGAN, text to image network, for birds generation.")
    parser.add_argument("mode", metavar="MODE", type=str,
                            help="-- mode of code running AttnGAN.\nOptions: train_DAMSM, train_AttnGAN, sample", )
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=-1)
    parser.add_argument('--data', dest='data_dir', type=str, default=None)
    parser.add_argument('--output', dest='output_dir', type=str, default=None)
    parser.add_argument('--state', dest='state_dict_path', type=str, default=None)
    parser.add_argument('--seed', dest="seed", type=int, help='manual seed')
    return parser.parse_args()


def update_configuration(args):
    # Set device parameters
    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id
        torch.cuda.set_device(cfg.GPU_ID)
        cudnn.benchmark = True

    running_shell_dir = os.getcwd()
    script_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # Set data directory
    if args.data_dir is not None:
        cfg.DATA_DIR = os.path.join(running_shell_dir, args.data_dir)
    else:
        cfg.DATA_DIR = os.path.join(script_dir, "data", "birds")

    # Set output directory
    if args.output_dir is not None:
        cfg.OUTPUT_DIR = os.path.join(running_shell_dir, args.output_dir)
    else:
        cfg.OUTPUT_DIR = os.path.join(script_dir, "output")

    # Set random seed
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if cfg.CUDA:
            torch.cuda.manual_seed_all(args.seed)

    # In case your training was interrupted
    if args.state_dict_path is not None:
        cfg.STATE_DICT = torch.load(os.path.join(running_shell_dir, args.state_dict_path))


def set_train_DAMSM_config():
    cfg.WORKERS = 6

    cfg.TREE.BRANCH_NUM = 1
    cfg.TREE.BASE_SIZE = 299

    cfg.TRAIN.FLAG = True
    cfg.TRAIN.BATCH_SIZE = 48
    cfg.TRAIN.MAX_EPOCH = 400
    cfg.TRAIN.SNAPSHOT_INTERVAL = 1
    cfg.TRAIN.ENCODER_LR = 0.00091
    cfg.TRAIN.RNN_GRAD_CLIP = 0.25

    cfg.TRAIN.SMOOTH.GAMMA1 = 4.0
    cfg.TRAIN.SMOOTH.GAMMA2 = 5.0
    cfg.TRAIN.SMOOTH.GAMMA3 = 10.0

    cfg.TEXT.EMBEDDING_DIM = 256
    cfg.TEXT.CAPTIONS_PER_IMAGE = 10


def set_train_AttnGAN_config():
    cfg.WORKERS = 4

    cfg.TREE.BRANCH_NUM = 3

    cfg.TRAIN.FLAG = True
    cfg.TRAIN.BATCH_SIZE = 10
    cfg.TRAIN.MAX_EPOCH = 600
    cfg.TRAIN.SNAPSHOT_INTERVAL = 1
    cfg.TRAIN.DISCRIMINATOR_LR = 0.0002
    cfg.TRAIN.GENERATOR_LR = 0.0002

    cfg.TRAIN.SMOOTH.GAMMA1 = 4.0
    cfg.TRAIN.SMOOTH.GAMMA2 = 5.0
    cfg.TRAIN.SMOOTH.GAMMA3 = 10.0
    cfg.TRAIN.SMOOTH.LAMBDA = 5.0

    cfg.GAN.DF_DIM = 64
    cfg.GAN.GF_DIM = 32
    cfg.GAN.Z_DIM = 100
    cfg.GAN.R_NUM = 2

    cfg.TEXT.EMBEDDING_DIM = 256
    cfg.CAPTIONS_PER_IMAGE = 10


def main():
    args = parse_args()

    update_configuration(args)

    if args.mode == "train_DAMSM":
        set_train_DAMSM_config()
        train_DAMSM.pipeline()
    elif args.mode == "train_AttnGAN":
        if args.state_dict_path is None:
            return ValueError("state should be passed")
        set_train_AttnGAN_config()
        run_AttnGAN.pipeline()
    else:
        raise ValueError("Choose one mode from ['train_DAMSM', 'train_AttnGAN', 'sample']")


if __name__ == "__main__":
    main()