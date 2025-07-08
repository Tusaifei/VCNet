# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import List, Tuple
from easydict import EasyDict

from torch import tensor, Tensor
import torch.nn as nn
from .utils import resize


class BaseSegmentor(nn.Module, metaclass=ABCMeta):
    """Base class for segmentors.

    Args:
        data_preprocessor (dict, optional): Model preprocessing config
            for processing the input data. it usually includes
            ``to_rgb``, ``pad_size_divisor``, ``pad_val``,
            ``mean`` and ``std``. Default to None.
       init_cfg (dict, optional): the config to control the
           initialization. Default to None.
    """

    def __init__(self,
                 data_preprocessor: EasyDict = None,
                 init_cfg: EasyDict = None):
        super().__init__()

    @property
    def with_neck(self) -> bool:
        """bool: whether the segmentor has neck"""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_auxiliary_head(self) -> bool:
        """bool: whether the segmentor has auxiliary head"""
        return hasattr(self,
                       'auxiliary_head') and self.auxiliary_head is not None

    @property
    def with_decode_head(self) -> bool:
        """bool: whether the segmentor has decode head"""
        return hasattr(self, 'decode_head') and self.decode_head is not None

    @abstractmethod
    def extract_feat(self, inputs: Tensor) -> bool:
        """Placeholder for extract features from images."""
        pass

    @abstractmethod
    def encode_decode(self, inputs: Tensor, batch_data_samples: List):
        """Placeholder for encode images with backbone and decode into a
        semantic segmentation map of the same size as input."""
        pass

    def forward(self,
                inputs: Tensor,
                data_samples: List = None,
                mode: str = 'tensor'):
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`SegDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape (N, C, ...) in
                general.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`. Default to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    @abstractmethod
    def loss(self, inputs: Tensor, data_samples: List) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        pass

    @abstractmethod
    def predict(self,
                inputs: Tensor,
                data_samples: List = None) -> List:
        """Predict results from a batch of inputs and data samples with post-
        processing."""
        pass

    @abstractmethod
    def _forward(self,
                 inputs: Tensor,
                 data_samples: List = None) -> Tuple[List[Tensor]]:
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass

    def postprocess_result(self,
                           seg_logits: Tensor,
                           data_samples: List = None) -> List:
        """ Convert results list to `SegDataSample`.
        Args:
            seg_logits (Tensor): The segmentation results, seg_logits from
                model of each input image.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`. Default to None.
        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
