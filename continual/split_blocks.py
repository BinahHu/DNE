import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import copy

class split_config:
    default_stack = True
    default_split = True
    def __init__(self):
        pass

    @staticmethod
    def set_stack(stack):
        split_config.default_stack = stack

    @staticmethod
    def set_split(split):
        split_config.default_split = split

class split_Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(split_Linear, self).__init__()

        self.in_features_list = [in_features]
        self.out_features_list = [out_features]
        self.bias = bias
        self.split = split_config.default_split
        self.stack = split_config.default_stack
        self.linear_list = nn.ModuleList([nn.Linear(in_features, out_features, bias=bias)])
        self.reset_parameters()

    @property
    def in_features(self):
        return sum(self.in_features_list)

    @property
    def out_features(self):
        return sum(self.out_features_list)

    @property
    def block_length(self):
        return len(self.in_features_list)

    @property
    def device(self):
        return self.linear_list[0].weight.device

    def freeze_split_old(self):
        for b in range(self.block_length-1):
            for p in self.linear_list[b].parameters():
                p.requires_grad = False

    def expand(self, in_features, out_features, fix_old = True, init_new = True, split = True):
        self.split = split
        if not split:
            assert self.block_length == 1, "If split is enabled, there can only be one block"
            self.in_features_list[0] += in_features
            self.out_features_list[0] += out_features
            new_linear = nn.Linear(self.in_features, self.out_features, bias=self.bias).to(self.device)
            if in_features > 0:
                new_linear.weight.data[:-out_features, :-in_features] = self.linear_list[-1].weight.data
            else:
                new_linear.weight.data[:-out_features, ...] = self.linear_list[-1].weight.data
            if self.bias:
                new_linear.bias.data[:-out_features] = self.linear_list[-1].bias.data
            self.linear_list[-1] = new_linear
            return

        self.in_features_list.append(in_features)
        self.out_features_list.append(out_features)
        if fix_old:
            self.freeze_split_old()
        if self.stack:
            extra_linear = nn.Linear(sum(self.in_features_list), out_features, bias=self.bias).to(self.device)
        else:
            extra_linear = nn.Linear(in_features, out_features, bias=self.bias).to(self.device)
        if init_new:
            extra_linear.apply(self._init_weights)
        self.linear_list.append(extra_linear)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def local_init_latest_v(self, dim):
        self.linear_list[-1].weight.data[-dim:, -dim:].copy_(torch.eye(dim))

    def local_init_latest_proj(self, num_heads, locality_strength=1.):
        locality_distance = 1  # max(1,1/locality_strength**.5)

        kernel_size = int(num_heads ** .5)
        center = (kernel_size - 1) / 2 if kernel_size % 2 == 0 else kernel_size // 2
        for h1 in range(kernel_size):
            for h2 in range(kernel_size):
                position = h1 + kernel_size * h2
                self.linear_list[-1].weight.data[position, 2] = -1
                self.linear_list[-1].weight.data[position, 1] = 2 * (h1 - center) * locality_distance
                self.linear_list[-1].weight.data[position, 0] = 2 * (h2 - center) * locality_distance
        self.linear_list[-1].weight.data *= locality_strength

    def reset_parameters(self, reset_type='new') -> None:
        if reset_type == 'new':
            b_range = [self.block_length - 1]
        elif reset_type == 'all':
            b_range = range(self.block_length)
        elif reset_type == 'old':
            b_range = range(self.block_length - 1)
        else:
            raise NotImplementedError

        for b in b_range:
            self.linear_list[b].reset_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        outs = []
        with torch.no_grad():
            for b in range(self.block_length - 1):
                if self.stack:
                    in_features = sum(self.in_features_list[:b+1])
                    outs.append(self.linear_list[b](input[..., :in_features]))
                else:
                    in_features_p1 = sum(self.in_features_list[:b])
                    in_features_p2 = sum(self.in_features_list[:b+1])
                    outs.append(self.linear_list[b](input[..., in_features_p1:in_features_p2]))
        if self.stack or (not self.split):
            outs.append(self.linear_list[-1](input))
        else:
            outs.append(self.linear_list[-1](input[..., sum(self.in_features_list[:-1]):]))
        return torch.cat(outs, dim=-1)

class split_Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        super(split_Conv2d, self).__init__()
        self.in_channels_list = [in_channels]
        self.out_channels_list = [out_channels]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        #For split conv2d, self.stack is always True
        self.stack = True
        self.split = split_config.default_split


        self.conv_list = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                                  padding=padding, dilation=dilation, groups=groups, bias=bias,
                                                  padding_mode=padding_mode)])
        self.reset_parameters()

    @property
    def in_channels(self):
        return sum(self.in_channels_list)

    @property
    def out_channels(self):
        return sum(self.out_channels_list)

    @property
    def block_length(self):
        return len(self.in_channels_list)

    @property
    def device(self):
        return self.conv_list[0].weight.device

    def reset_parameters(self, reset_type='new') -> None:
        if reset_type == 'new':
            b_range = [self.block_length - 1]
        elif reset_type == 'all':
            b_range = range(self.block_length)
        elif reset_type == 'old':
            b_range = range(self.block_length - 1)
        else:
            raise NotImplementedError
        for b in b_range:
            self.conv_list[b].reset_parameters()

    def freeze_split_old(self):
        for b in range(self.block_length-1):
            for p in self.conv_list[b].parameters():
                p.requires_grad = False

    def expand(self, in_channels, out_channels, fix_old = True, init_new = True, split = True):
        self.split = split
        if not split:
            assert self.block_length == 1, "If split is enabled, there can only be one block"
            self.in_channels_list[0] += in_channels
            self.out_channels_list[0] += out_channels
            new_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size,
                                        stride=self.stride, padding=self.padding, dilation=self.dilation,
                                        groups=self.groups, bias=self.bias, padding_mode=self.padding_mode).to(self.device)
            if in_channels > 0:
                new_conv.weight.data[:-out_channels, :-in_channels, ...] = self.conv_list[-1].weight.data
            else:
                new_conv.weight.data[:-out_channels, ...] = self.conv_list[-1].weight.data
            if self.bias:
                new_conv.bias.data[:-out_channels] = self.conv_list[-1].bias.data
            self.conv_list[-1] = new_conv
            return

        self.in_channels_list.append(in_channels)
        self.out_channels_list.append(out_channels)
        if fix_old:
            self.freeze_split_old()
        if self.stack:
            extra_conv = nn.Conv2d(sum(self.in_channels_list), out_channels, kernel_size=self.kernel_size,
                                            stride=self.stride, padding=self.padding, dilation=self.dilation,
                                            groups=self.groups, bias=self.bias, padding_mode=self.padding_mode).to(self.device)
        else:
            extra_conv = nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size,
                                   stride=self.stride, padding=self.padding, dilation=self.dilation,
                                   groups=self.groups, bias=self.bias, padding_mode=self.padding_mode).to(self.device)
        if init_new:
            extra_conv.apply(self._init_weights)
        self.conv_list.append(extra_conv)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        outs = []
        with torch.no_grad():
            for b in range(self.block_length - 1):
                if self.stack:
                    in_channels = sum(self.in_channels_list[:b + 1])
                    outs.append(self.conv_list[b](input[:, :in_channels, ...]))
                else:
                    in_channels_p1 = sum(self.in_channels_list[:b])
                    in_channels_p2 = sum(self.in_channels_list[:b+1])
                    outs.append(self.conv_list[b](input[:, in_channels_p1:in_channels_p2, ...]))
        if self.stack or (not self.split):
            outs.append(self.conv_list[-1](input))
        else:
            outs.append(self.conv_list[-1](input[:, sum(self.in_channels_list[:-1]):, ...]))
        return torch.cat(outs, dim=1)

class split_LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super(split_LayerNorm, self).__init__()
        self.shape_list = [normalized_shape]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.norm_layers = nn.ModuleList([nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)])
        self.reset_parameters()

    @property
    def block_length(self):
        return len(self.shape_list)

    @property
    def device(self):
        return self.norm_layers[0].weight.device

    def reset_parameters(self, reset_type='new') -> None:
        if reset_type == 'new':
            b_range = [self.block_length - 1]
        elif reset_type == 'all':
            b_range = range(self.block_length)
        elif reset_type == 'old':
            b_range = range(self.block_length - 1)
        else:
            raise NotImplementedError
        for b in b_range:
            self.norm_layers[b].reset_parameters()

    def freeze_split_old(self):
        for b in range(self.block_length-1):
            for p in self.norm_layers[b].parameters():
                p.requires_grad = False
            self.norm_layers[b].eval()

    def expand(self, normalized_shape, fix_old = True, init_new = True, split = True):
        if not split:
            assert self.block_length == 1, "If split is enabled, there can only be one block"
            self.shape_list[0] += normalized_shape
            new_norm = nn.LayerNorm(sum(self.shape_list), eps=self.eps, elementwise_affine=self.elementwise_affine).to(self.device)
            new_norm.weight.data[:-normalized_shape] = self.norm_layers[-1].weight.data
            new_norm.bias.data[:-normalized_shape] = self.norm_layers[-1].bias.data
            self.norm_layers[-1] = new_norm
            return

        self.shape_list.append(normalized_shape)
        if fix_old:
            self.freeze_split_old()
        extra_norm = nn.LayerNorm(normalized_shape, eps=self.eps, elementwise_affine=self.elementwise_affine).to(self.device)
        if init_new:
            extra_norm.apply(self._init_weights)
        self.norm_layers.append(extra_norm)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        outs = []
        with torch.no_grad():
            for b in range(self.block_length - 1):
                v1 = sum(self.shape_list[:b])
                v2 = sum(self.shape_list[:b+1])
                outs.append(self.norm_layers[b](input[..., v1:v2]))
        v = sum(self.shape_list[:-1])
        outs.append(self.norm_layers[-1](input[..., v:]))
        return torch.cat(outs, dim=-1)

class split_Dropout(nn.Module):
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super(split_Dropout, self).__init__()
        pass
        # Not sure if we need to implement it