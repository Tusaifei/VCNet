import argparse
import rasterio
from rasterio.errors import RasterioIOError
from torch.utils.data.dataset import Dataset
import os
import random
from easydict import EasyDict
import pandas as pd
import numpy as np
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.encoder_decoder import EncoderDecoder
import torch.nn.functional as F
import utils 
import torch
from networks.vit_seg_modeling_L2HNet import L2HNet

class TileInferenceDataset(Dataset):
    
    def __init__(self, fn, chip_size, stride, transform=None, windowed_sampling=False, verbose=False):
        self.fn = fn
        self.chip_size = chip_size
        
        self.transform = transform
        self.windowed_sampling = windowed_sampling
        self.verbose = verbose
        
        with rasterio.open(self.fn) as f:
            height, width = f.height, f.width
            self.num_channels = f.count
            self.dtype = f.profile["dtype"]
            if not windowed_sampling: # if we aren't using windowed sampling, then go ahead and read in all of the data
                self.data = np.rollaxis(f.read(), 0, 3)
            
        self.chip_coordinates = [] # upper left coordinate (y,x), of each chip that this Dataset will return
        for y in list(range(0, height - self.chip_size, stride)) + [height - self.chip_size]:
            for x in list(range(0, width - self.chip_size, stride)) + [width - self.chip_size]:
                self.chip_coordinates.append((y,x))
        self.num_chips = len(self.chip_coordinates)

        if self.verbose:
            print("Constructed TileInferenceDataset -- we have %d by %d file with %d channels with a dtype of %s. We are sampling %d chips from it." % (
                height, width, self.num_channels, self.dtype, self.num_chips
            ))
            
    def __getitem__(self, idx):
        y, x = self.chip_coordinates[idx]
        
        if self.windowed_sampling:
            try:
                with rasterio.Env():
                    with rasterio.open(self.fn) as f:
                        img = np.rollaxis(f.read(window=rasterio.windows.Window(x, y, self.chip_size, self.chip_size)), 0, 3)
            except RasterioIOError as e: # NOTE(caleb): I put this here to catch weird errors that I was seeing occasionally when trying to read from COGS - I don't remember the details though
                print("Reading %d failed, returning 0's" % (idx))
                img = np.zeros((self.chip_size, self.chip_size, self.num_channels), dtype=np.uint8)
        else:
            img = self.data[y:y+self.chip_size, x:x+self.chip_size]


        if self.transform is not None:
            img = self.transform(img)

        return img, np.array((y,x))
        
    def __len__(self):
        return self.num_chips

parser = argparse.ArgumentParser()
CHIP_SIZE = 224
PADDING = 112
assert PADDING % 2 == 0
HALF_PADDING = PADDING//2
CHIP_STRIDE = CHIP_SIZE - PADDING
parser.add_argument('--dataset', type=str, default='Chesapeake', help='experiment_name')
parser.add_argument('--max_epochs', type=int, default=30, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--CNN_width', type=int, default=64, help='L2HNet_width_size, default is 64: light mode. Set to 128: normal mode')
parser.add_argument('--save_path', type=str)
parser.add_argument('--model_path', type=str)
parser.add_argument('--gpu', type=str, help='Select GPU number to train' )
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

backbone = EasyDict(dict(
    type='ViTCoMer',
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
    window_attn=[
        False, False, False, False, False, False, False, False, False,
        False, False, False
    ],
    window_size=[
        None, None, None, None, None, None, None, None, None, None, None,
        None
    ]))

decode_head = EasyDict(dict(
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
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)))

auxiliary_head = EasyDict(dict(
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
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)))

def inference(args, model, test_save_path=None):
    model.eval()
    input_dataframe = pd.read_csv(args.list_dir)
    image_fns = input_dataframe["image_fn"].values

    for image_idx in range(len(image_fns)):
        image_fn = image_fns[image_idx]

        print("(%d/%d) Processing %s" % (image_idx, len(image_fns), image_fn), end=" ... ")
        #-------------------
        # Load input and create dataloader
        #-------------------
        def image_transforms(img):
            img = (img - utils.IMAGE_MEANS) / utils.IMAGE_STDS
            img = np.rollaxis(img, 2, 0).astype(np.float32)
            img = torch.from_numpy(img)
            return img

        with rasterio.open(image_fn) as f:
            input_width, input_height = f.width, f.height
            input_profile = f.profile.copy()
        
        dataset = TileInferenceDataset(image_fn, chip_size=CHIP_SIZE, stride=CHIP_STRIDE, transform=image_transforms, verbose=False)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=16,
            num_workers=4,
            pin_memory=True,
        )

        #-------------------
        # Run model and organize output
        #-------------------

        output = np.zeros((args.num_classes, input_height, input_width), dtype=np.float32)
        kernel = np.ones((CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
        kernel[HALF_PADDING:-HALF_PADDING, HALF_PADDING:-HALF_PADDING] = 5
        counts = np.zeros((input_height, input_width), dtype=np.float32)

        for i, (data, coords) in enumerate(dataloader):
            data = data.cuda()
            with torch.no_grad():
                comer_out, cnn_out = model(data)
                t_output = F.softmax((comer_out+cnn_out)/2.0, dim=1).cpu().numpy()

            for j in range(t_output.shape[0]):
                y, x =  coords[j]

                output[:, y:y+CHIP_SIZE, x:x+CHIP_SIZE] += t_output[j] * kernel
                counts[y:y+CHIP_SIZE, x:x+CHIP_SIZE] += kernel
        
        output = output / counts
        output_hard = output.argmax(axis=0).astype(np.uint8)

        #-------------------
        # Save output
        #-------------------
        output_profile = input_profile.copy()
        output_profile["driver"] = "GTiff"
        output_profile["dtype"] = "uint8"
        output_profile["count"] = 1
        output_profile["nodata"] = 0

        output_fn = image_fn.split("/")[-1] 
        output_fn = output_fn.replace("naip", "predictions") # name the predictions
        output_fn = os.path.join(test_save_path, output_fn)

        with rasterio.open(output_fn, "w", **output_profile) as f:
            f.write(output_hard, 1)
            f.write_colormap(1, utils.LABEL_IDX_COLORMAP)   
    return "Testing Finished!"


if __name__ == "__main__":
    cudnn.benchmark = True
    cudnn.deterministic = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'Chesapeake': {
            'list_dir': './dataset/CSV_list/Chesapeake_NewYork.csv', # The path of the *.csv file
            'num_classes': 17
        }
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    vit_patches_size=16
    net = EncoderDecoder(
        backbone=backbone,
        decode_head=decode_head,
        auxiliary_head=auxiliary_head,
        cnn_encoder=L2HNet(width=args.CNN_width),
        pretrained=None,
    ).cuda(device=0) 
    snapshot=args.model_path 
    net.load_state_dict(torch.load(snapshot))
    test_save_path=args.save_path 
    os.makedirs(test_save_path, exist_ok=True)

    inference(args, net, test_save_path)


