# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import List, Optional
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Conv2d
from torch.nn.modules.utils import _pair
from torch import Tensor
from .base import BaseSegmentor
from .vit_FuseX import ViTFuseX
from .uper_head import UPerHead
from .fcn_head import FCNHead
from .vit_seg_modeling import DecoderCup, SegmentationHead


class EncoderDecoder(BaseSegmentor):
    def __init__(self,
                 backbone: EasyDict = None,
                 decode_head: EasyDict = None,
                 neck: EasyDict = None,
                 auxiliary_head: EasyDict = None,
                 train_cfg: EasyDict = None,
                 test_cfg: EasyDict = None,
                 data_preprocessor: EasyDict = None,
                 pretrained: EasyDict = None,
                 cnn_encoder: nn.Module = None,
                 init_cfg: EasyDict = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        '''
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        '''
        backbone.pretrained = pretrained
        self.backbone = ViTFuseX(num_heads=backbone.num_heads, 
                                 conv_inplane=backbone.conv_inplane, 
                                 n_points=backbone.n_points, 
                                 deform_num_heads=backbone.deform_num_heads,
                                 interaction_indexes=backbone.interaction_indexes, 
                                 cffn_ratio=backbone.cffn_ratio,
                                 deform_ratio=backbone.deform_ratio,
                                 in_chans=backbone.in_channel,
                                 embed_dim=backbone.embed_dim,
                                 pretrained=backbone.pretrained)
        self.cnn_encoder = cnn_encoder
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

        # for additional cnn
        img_size = _pair(224)
        grid_size = (14, 14)
        hidden_size = 192
        patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
        patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
        n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
        self.patch_embeddings = Conv2d(
            in_channels=1024,
            out_channels=1,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, hidden_size))
        self.dropout = Dropout(0.1)
        self.decoder_cup = DecoderCup(EasyDict({
            'hidden_size': hidden_size
        }))
        self.segmentation_head = SegmentationHead(
            in_channels=256,
            out_channels=decode_head.num_classes,
            kernel_size=3
        )

        self.adjust_channel_conv = nn.Conv2d(in_channels=1024, out_channels=192, kernel_size=1, stride=1, padding=0)
        self.adjust_channel_conv2 = nn.Conv2d(in_channels=384, out_channels=192, kernel_size=1, stride=1, padding=0)


    def _init_decode_head(self, decode_head) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = UPerHead(
            in_channels=decode_head.in_channels,
            in_index=decode_head.in_index,
            pool_scales=decode_head.pool_scales,
            channels=decode_head.channels,
            dropout_ratio=decode_head.dropout_ratio,
            num_classes=decode_head.num_classes,
            norm_cfg=decode_head.norm_cfg,
            align_corners=decode_head.align_corners,
            loss_decode=decode_head.loss_decode
        )
        self.align_corners = decode_head.align_corners
        self.num_classes = decode_head.num_classes
        self.out_channels = None # decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head) -> None:
        # print(type(self))
        """Initialize ``auxiliary_head``"""
        # self.with_auxiliary_head = True
        self.auxiliary_head = FCNHead(
            in_channels=auxiliary_head.in_channels,
            in_index=auxiliary_head.in_index,
            channels=auxiliary_head.channels,
            num_convs=auxiliary_head.num_convs,
            concat_input=auxiliary_head.concat_input,
            dropout_ratio=auxiliary_head.dropout_ratio,
            num_classes=auxiliary_head.num_classes,
            norm_cfg=auxiliary_head.norm_cfg,
            align_corners=auxiliary_head.align_corners,
            loss_decode=auxiliary_head.loss_decode
        )

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(inputs)
        seg_logits = self.decode_head.predict(x, batch_img_metas,
                                              self.test_cfg)

        return seg_logits

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: List) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.loss(inputs, data_samples,
                                            self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(self, inputs: List[Tensor],
                                      data_samples: List) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def loss(self, inputs: Tensor, data_samples: List) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(inputs)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: List = None) -> List:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        seg_logits = self.inference(inputs, batch_img_metas)

        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: List = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
       

        y, features = self.cnn_encoder(inputs)

        features_for_cnn = features[0]
        y = self.patch_embeddings(y)
        y = y.flatten(2)
        y = y.transpose(-1, -2)
        embeddings = y + self.position_embeddings
        embeddings_for_vit = self.dropout(embeddings)
       
        x_FuseX = self.backbone((embeddings_for_vit, features_for_cnn))
        
        _x_cnn_tmp, x_cnn = self.decoder_cup(embeddings_for_vit, features)

        # merging feature from cnn
        x_FuseX[0] = self.adjust_channel_conv2(torch.cat([self.adjust_channel_conv(features[2]), x_FuseX[0]], dim=1))

        decoded2 = self.segmentation_head(x_cnn)
        decoded1 = self.decode_head.forward(x_FuseX)
 
        return decoded1, decoded2

    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        seg_logits = self.encode_decode(inputs, batch_img_metas)

        return seg_logits

    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole'], \
            f'Only "slide" or "whole" test mode are supported, but got ' \
            f'{self.test_cfg["mode"]}.'
        ori_shape = batch_img_metas[0]['ori_shape']
        if not all(_['ori_shape'] == ori_shape for _ in batch_img_metas):
            print_log(
                'Image shapes are different in the batch.',
                logger='current',
                level=logging.WARN)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas)

        return seg_logit

    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(inputs[0], batch_img_metas[0], rescale)
        for i in range(1, len(inputs)):
            cur_seg_logit = self.inference(inputs[i], batch_img_metas[i],
                                           rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred


if __name__ == '__main__':
    backbone = EasyDict(dict(
        type='ViTFuseX',
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
        num_classes=6,
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
        num_classes=6,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)))
    
