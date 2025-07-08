import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import copy


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class ConvModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 inplace=True,
                 with_spectral_norm=False,
                 padding_mode='zeros',
                 order=('conv', 'norm', 'act')):
        super(ConvModule, self).__init__()
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.padding_mode = padding_mode
        self.order = order

        # Build convolution layer
        self.conv = self.build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias
        )

        # Build normalization layer
        if norm_cfg is not None:
            self.norm = self.build_norm_layer(norm_cfg, out_channels)
        else:
            self.norm = None

        # Build activation layer
        if act_cfg['type'] == 'ReLU':
            self.act = nn.ReLU(inplace=inplace)
        else:
            raise NotImplementedError

        # Spectral normalization
        if with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

    def build_conv_layer(self, conv_cfg, *args, **kwargs):
        if conv_cfg is None:
            return nn.Conv2d(*args, **kwargs)
        # Add custom conv layer building logic here if needed
        return nn.Conv2d(*args, **kwargs)

    def build_norm_layer(self, norm_cfg, num_features):
        if norm_cfg['type'] == 'BN':
            return nn.BatchNorm2d(num_features)
        # Add custom norm layer building logic here if needed
        return nn.BatchNorm2d(num_features)

    def forward(self, x):
        if self.order == ('conv', 'norm', 'act'):
            x = self.conv(x)
            if self.norm is not None:
                x = self.norm(x)
            x = self.act(x)
        elif self.order == ('act', 'conv', 'norm'):
            x = self.act(x)
            x = self.conv(x)
            if self.norm is not None:
                x = self.norm(x)
        else:
            raise ValueError('Invalid order')
        return x


class BaseModule(nn.Module):
    def __init__(self, init_cfg=None):
        super(BaseModule, self).__init__()
        self.init_cfg = copy.deepcopy(init_cfg)
        self._is_init = False

    def init_weights(self):
        if not self._is_init:
            if self.init_cfg is not None:
                # 这里可以添加自定义的初始化逻辑，比如初始化权重
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
            self._is_init = True

    def forward(self, x):
        # 必须实现前向传播
        raise NotImplementedError