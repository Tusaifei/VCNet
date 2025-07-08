import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.encoder_decoder import EncoderDecoder
from trainer import trainer_dataset
from networks.vit_seg_modeling_L2HNet import L2HNet
from easydict import EasyDict

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Chesapeake', help='experiment_name')
parser.add_argument('--max_epochs', type=int, default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--CNN_width', type=int, default=64, help='L2HNet_width_size, default is 64: light mode. Set to 128: normal mode')
parser.add_argument('--savepath', type=str, default='/home/tsf/VCNet/vcnet-train1')
parser.add_argument('--gpu', type=str, default='0', help='Select GPU number to train' )
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# 规范命名
backbone_config = EasyDict(dict(
    type='ViTFuseX',
    in_channel=4,
    patch_size=16,
    embed_dim=192,
    depth=12,
    num_heads=3,
    mlp_ratio=4,
    drop_path_rate=0.1,
    conv_inplane=64,
    n_points=4,
    deform_num_heads=6,
    cffn_ratio=0.25,
    deform_ratio=1.0,
    interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
    window_attn=[False] * 12,
    window_size=[None] * 12
))

decode_head_config = EasyDict(dict(
    type='UPerHead',
    in_channels=[192, 192, 192, 192],
    in_index=[0, 1, 2, 3],
    pool_scales=(1, 2, 3, 6),
    channels=512,
    dropout_ratio=0.1,
    num_classes=5,
    norm_cfg=dict(type='SyncBN', requires_grad=True),
    align_corners=False,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
))

auxiliary_head_config = EasyDict(dict(
    type='FCNHead',
    in_channels=192,
    in_index=2,
    channels=256,
    num_convs=1,
    concat_input=False,
    dropout_ratio=0.1,
    num_classes=5,
    norm_cfg=dict(type='SyncBN', requires_grad=True),
    align_corners=False,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
))

if __name__ == "__main__":
    cudnn.benchmark = True
    cudnn.deterministic = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Chesapeake': {
            'list_dir': '/home/tsf/VCNet/dataset/CSV_list/Chesapeake_NewYork.csv',
            'num_classes': 17
        }
    }
    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    snapshot_path = args.savepath 
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    net = EncoderDecoder(
        backbone=backbone_config,
        decode_head=decode_head_config,
        auxiliary_head=auxiliary_head_config,
        cnn_encoder = L2HNet(width=args.CNN_width),
        pretrained = "/home/tsf/VCNet/deit_tiny_patch16_224-a1311bcf.pth"
    ).cuda(device=0) 

    trainer_dataset(args, net, snapshot_path)