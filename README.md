# VCNet - a weakly supervised end-to-end deep-learning network for large-scale HR land-cover mapping 

## Environment Requirements  
### Hardware Requirements  
- NVIDIA GPU (recommended video memory ≥ 8GB)  
- System memory ≥ 16GB  
- Storage: Approximately 20GB for datasets and models  

### Software Dependencies  
```bash
torch==1.4.0  
torchvision==0.5.0  
numpy>=1.19.5  
tqdm>=4.62.3  
tensorboard>=2.7.0  
tensorboardX>=2.5.1  
ml-collections>=0.1.0  
medpy>=0.4.0  
SimpleITK>=2.1.1  
scipy>=1.7.3  
h5py>=3.6.0  
rasterio==1.2.10  
easydict>=1.9  
```  


## Dataset Preparation  
### Supported Datasets  
 The Chesapeake Bay Dataset contains 1-meter resolution images and a 30-meter resolution land-cover product as the training data pairs and also contains a 1-meter resolution ground reference for assessment.

### Data Structure  
```
dataset/
├── CSV_list/
│   ├── Chesapeake_NewYork.csv    # Training data list
│   └── ...
└── imagery/
    ├── HR image/    # Training image folder
    ├── LR label/     # Training label folder 
    └── CCLC/     # Validation data folder
```  

### Configuration Instructions  
Modify the following parameters in `train.py`:  
- `--dataset`: Specify the dataset name (e.g., `Chesapeake`)  
- `--list_dir`: Point to the CSV list file path  


## Code Structure  
```
VCNet/
├── train.py              # Main training script
├── test.py               # Testing and inference script
├── auto_test.sh          # Batch testing script
├── networks/             # Network model definitions
│   ├── vit_seg_modeling.py       # ViT_FuseX model implementation
│   ├── vit_seg_modeling_L2HNet.py # L2HNet model implementation
│   └── encoder_decoder.py        # Dual-model concatenation architecture
├── trainer.py            # Trainer implementation
├── mIoU.py               # accuracy validation
├── requirements.txt      # Dependency list
└── README.md             # Project documentation
```  


## Training and Testing  
### Start Training  
```bash
python train.py \
--dataset Chesapeake \
--batch_size 2 \
--max_epochs 100 \
--savepath ./checkpoints \
--gpu 0
```  

### Key Parameter Explanations  
- `--batch_size`: Adjust based on GPU video memory (recommended 2-8)  
- `--base_lr`: Base learning rate (default 0.01)  
- `--CNN_width`: L2HNet width (64 for lightweight mode, 128 for standard mode)  

### Model Testing  
```bash
python test.py \
--checkpoint ./checkpoints/model.pth \
--dataset Chesapeake \
--gpu 0
```  


## Model Architecture  
### Core Components  
1. **ViT_FuseX Encoder**: Captures global semantic information with multi-scale features of remote sensing images  
2. **L2HNet Encoder**: Extracts high resolution spatial features from HR images
3. **Feature Fusion Module**: Fuses features from dual branches to enhance segmentation accuracy  

### Concatenation Method  
The dual-model concatenation is implemented via the `EncoderDecoder` class. Code example:  
```python
net = EncoderDecoder(
    backbone=backbone_config,
    decode_head=decode_head_config,
    auxiliary_head=auxiliary_head_config,
    cnn_encoder=L2HNet(width=args.CNN_width),
    pretrained="/path/to/pretrained_model.pth"
).cuda()
```