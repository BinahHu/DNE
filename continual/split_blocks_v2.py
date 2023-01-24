import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import copy

class Proj_wAttn(nn.Module):
    def __init__(self, head_num, head_dim, out_head_dim=None, attn_thd=0, proj_type="Linear", self_attn=True, extra_heads=1):
        super(Proj_wAttn, self).__init__()
        self.head_num = head_num
        self.extra_heads = extra_heads
        self.head_dim = head_dim
        self.out_head_dim = out_head_dim or head_dim
        self.fixed_attn = False
        self.attn = None
        self.attn_thd = attn_thd
        self.proj_type = proj_type
        self.self_attn = self_attn
        self.proj_blocks = nn.ModuleList()
        if proj_type == "Linear":
            for i in range(head_num):
                self.proj_blocks.append(nn.Linear(head_dim, self.out_head_dim))
            if self.self_attn:
                for i in range(self.extra_heads):
                    self.proj_blocks.append(nn.Linear(head_dim, self.out_head_dim))
        elif proj_type == "shared_Linear":
            self.proj_blocks = nn.Linear(head_dim, self.out_head_dim)
        else:
            raise NotImplementedError(f'Unknown projection type {proj_type}')

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def reset_parameters(self):
        self.apply(self._init_weights)

    def purage(self):
        assert self.fixed_attn
        assert self.proj_type != "shared_Linear"
        attn_mask = (self.attn.max(dim=2)[0] > self.attn_thd).squeeze()
        for i in range(attn_mask.shape[0]):
            if not attn_mask[i]:
                self.proj_blocks[i] = nn.Identity()

    def set_fixed_attn(self, attn):
        if attn is not None:
            self.attn = nn.Parameter(attn.detach().unsqueeze(0).unsqueeze(0))
            self.fixed_attn = True

    def forward(self, x, attn=None):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        B, N, C = x.shape
        true_head_num = self.head_num + (self.extra_heads if self.self_attn else 0)
        assert self.head_dim * true_head_num == C
        x = x.reshape(B, N, true_head_num, self.head_dim).permute(2, 0, 1, 3)

        if self.fixed_attn or attn is None:
            attn_mask = (self.attn.max(dim=2)[0] > self.attn_thd)
            forward_attn = self.attn.expand(B, N, -1, -1)
        else:
            attn_mask = torch.ones(true_head_num, dtype=torch.bool).cuda()
            forward_attn = attn
        if self.proj_type == "shared_Linear":
            x_proj = self.proj_blocks(x)
            x_proj = x_proj[attn_mask,:,:,:].permute(1, 2, 0, 3)
        else:
            x_proj = []
            for i in range(true_head_num):
                if attn_mask[i]:
                    x_proj.append(self.proj_blocks[i](x[i]))
            x_proj = torch.cat(x_proj, dim=-1).reshape(B, N, true_head_num, self.out_head_dim)
        x_proj = (forward_attn[:,:,:,attn_mask] @ x_proj).reshape(B, N, -1)
        return x_proj

class Task_Attn(nn.Module):
    def __init__(self, head_dim, qk_scale=None, bias=False, attn_drop=0., record_mean=True, momentum=0.9,
                 naive_mean=False, constant_scaled=False, self_attn=True, lambda_scaled=True, lambda_scaled_init=12,
                 q_head_num=-1, k_head_num=-1):
        super(Task_Attn, self).__init__()
        self.head_dim = head_dim
        self.naive_mean = naive_mean
        self.scaled = constant_scaled
        self.lambda_scaled = lambda_scaled
        self.lambda_scaled_init = lambda_scaled_init
        if self.lambda_scaled:
            self.lambda_factor = nn.Parameter(torch.ones(1).cuda() * lambda_scaled_init)

        self.record_mean = record_mean and True
        self.mean_attn = None
        self.momentun = momentum

        if naive_mean:
            return
        self.self_attn = self_attn
        self.scale = qk_scale or head_dim ** -0.5
        self.q_head_num = q_head_num
        self.k_head_num = k_head_num
        if self.self_attn:
            self.k_head_num += self.q_head_num
        if self.k_head_num > 0:
            self.k = nn.ModuleList()
            for i in range(self.k_head_num):
                self.k.append(nn.Linear(head_dim, head_dim, bias=bias))
        else:
            self.k = nn.Linear(head_dim, head_dim, bias=bias)
        if self.q_head_num > 0:
            self.q = nn.ModuleList()
            for i in range(self.q_head_num):
                self.q.append(nn.Linear(head_dim, head_dim, bias=bias))
        else:
            self.q = nn.Linear(head_dim, head_dim, bias=bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.apply(self._init_weights)

    def reset_parameters(self):
        self.apply(self._init_weights)
        self.mean_attn = None
        if self.lambda_scaled:
            self.lambda_factor = nn.Parameter(torch.ones(1).cuda() * self.lambda_scaled_init)

    def update_mean_attn(self, attn):
        B, N, H1, H2 = attn.shape
        attn = attn.reshape(B*N, H1, H2).mean(dim=0).detach()
        if self.mean_attn is None:
            self.mean_attn = torch.ones(H1, H2).cuda() / (1 if self.constant_scaled else H2)
        self.mean_attn = (1 - self.momentun) * attn + self.momentun * self.mean_attn

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_old, x_new):
        if self.naive_mean:
            if len(x_new.shape) == 2:
                x_new = x_new.unsqueeze(1)
            B, N, C1 = x_new.shape
            H1 = C1 // self.head_dim

            if len(x_old.shape) == 2:
                x_old = x_old.unsqueeze(1)
            B, N, C2 = x_old.shape
            H2 = C2 // self.head_dim
            attn = torch.ones(B, N, H1, H2).cuda()
            if not self.constant_scaled:
                attn = attn.softmax(dim=-1)
            if self.mean_attn is None:
                self.mean_attn = attn.reshape(B*N, H1, H2).mean(dim=0)
            return attn


        if self.self_attn:
            x_old = torch.cat([x_old, x_new], dim=-1)

        if len(x_new.shape) == 2:
            x_new = x_new.unsqueeze(1)

        B, N, C = x_new.shape
        H = C // self.head_dim

        if self.q_head_num > 0:
            x_new = x_new.reshape(B, N, H, self.head_dim).permute(2, 0, 1, 3)
            q = []
            for i in range(self.q_head_num):
                q.append(self.q[i](x_new[i]))
            q = torch.cat(q, dim=-1).reshape(B, N, H, self.head_dim)
        else:
            x_new = x_new.reshape(B, N, H, self.head_dim)
            q = self.q(x_new)
        q = q * self.scale

        if len(x_old.shape) == 2:
            x_old = x_old.unsqueeze(1)

        B, N, C = x_old.shape
        H = C // self.head_dim
        if self.k_head_num > 0:
            x_old = x_old.reshape(B, N, H, self.head_dim).permute(2, 0, 1, 3)
            k = []
            for i in range(self.k_head_num):
                k.append(self.k[i](x_old[i]))
            k = torch.cat(k, dim=-1).reshape(B, N, H, self.head_dim)
        else:
            x_old = x_old.reshape(B, N, H, self.head_dim)
            k = self.k(x_old)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        if self.lambda_scaled:
            attn = attn * self.lambda_factor
        elif self.constant_scaled:
            attn = attn * H
        if self.record_mean:
            self.update_mean_attn(attn)
        attn = self.attn_drop(attn)

        return attn

class split_Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool=True, split=True, stack=True, simple_proj=False,
                 proj_type="Linear", fix_attn=False, head_dim=32, attn_thd=0, out_head_dim=None, self_attn=True,
                 attn_qk_linear=False) -> None:
        super(split_Linear, self).__init__()

        self.in_features_list = [in_features]
        self.out_features_list = [out_features]
        self.bias = bias

        self.split = split
        self.stack = stack

        self.linear_list = nn.ModuleList([nn.Linear(in_features, out_features, bias=bias)])

        self.simple_proj = simple_proj

        self.head_num = in_features // head_dim
        self.head_dim = head_dim
        self.out_head_dim = out_head_dim or head_dim
        self.fix_attn = fix_attn
        self.attn_thd = attn_thd
        self.proj_type = proj_type
        self.self_attn = self_attn
        self.attn_qk_linear = attn_qk_linear

        if not self.simple_proj:
            self.proj_blocks = nn.ModuleList([Proj_wAttn(1, 1, attn_thd=attn_thd, proj_type=proj_type,
                                                        out_head_dim=1)])
            self.curr_attn_block = Task_Attn(1, record_mean=self.fix_attn, lambda_scaled_init=12)
            # self.proj_blocks = nn.ModuleList(
            #     [Proj_wAttn(self.head_num, self.head_dim, attn_thd=attn_thd, proj_type=proj_type,
            #                 out_head_dim=self.out_head_dim)])
            # self.curr_attn_block = Task_Attn(self.head_dim, record_mean=self.fix_attn)
        if (not self.fix_attn) and (not self.simple_proj):
            self.attn_blocks = nn.ModuleList()
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
            if not self.simple_proj:
                for p in self.proj_blocks[b].parameters():
                    p.requires_grad = False
                if not self.fix_attn:
                    for p in self.attn_blocks[b].parameters():
                        p.requires_grad = False

    def fix_and_update_attn(self):
        if self.simple_proj:
            return
        if self.fix_attn:
            self.proj_blocks[-1].set_fixed_attn(self.curr_attn_block.mean_attn)
        else:
            self.attn_blocks.append(self.curr_attn_block)

    def expand(self, in_features, out_features, fix_old=True):
        if not self.split:
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

        head_num = self.in_features // self.head_dim
        extra_heads = in_features // self.head_dim
        self.in_features_list.append(in_features)
        self.out_features_list.append(out_features)
        if fix_old:
            self.freeze_split_old()
        if self.simple_proj:
            if self.stack:
                extra_linear = nn.Linear(sum(self.in_features_list), out_features, bias=self.bias).to(self.device)
            else:
                extra_linear = nn.Linear(in_features, out_features, bias=self.bias).to(self.device)
            extra_linear.apply(self._init_weights)
            self.linear_list.append(extra_linear)
        else:
            extra_linear = nn.Linear(in_features, out_features, bias=self.bias).to(self.device)
            extra_linear.apply(self._init_weights)
            self.linear_list.append(extra_linear)

            extra_proj = Proj_wAttn(head_num, self.head_dim, out_head_dim=self.out_head_dim, attn_thd=self.attn_thd,
                                    proj_type=self.proj_type, self_attn=self.self_attn, extra_heads=extra_heads).to(self.device)
            self.proj_blocks.append(extra_proj)

            if self.fix_attn:
                self.curr_attn_block.reset_parameters()
            else:
                if self.attn_qk_linear:
                    self.curr_attn_block = Task_Attn(self.head_dim, record_mean=self.fix_attn,
                                                     self_attn=self.self_attn, lambda_scaled_init=head_num,
                                                     q_head_num=extra_heads, k_head_num=head_num).to(self.device)
                else:
                    self.curr_attn_block = Task_Attn(self.head_dim, record_mean=self.fix_attn,
                                                     self_attn=self.self_attn, lambda_scaled_init=head_num).to(self.device)

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
            if b < len(self.linear_list):
                self.linear_list[b].reset_parameters()
            if not self.simple_proj:
                self.proj_blocks[b].reset_parameters()
                if not self.fix_attn:
                    if b < len(self.attn_blocks):
                        self.attn_blocks[b].reset_parameters()

    def forward(self, input: torch.Tensor, debug=False) -> torch.Tensor:
        outs = []
        with torch.no_grad():
            for b in range(self.block_length - 1):
                if self.simple_proj:
                    if self.stack:
                        curr_features = sum(self.in_features_list[:b+1])
                        outs.append(self.linear_list[b](input[..., :curr_features]))
                    else:
                        in_features_p1 = sum(self.in_features_list[:b])
                        in_features_p2 = sum(self.in_features_list[:b + 1])
                        outs.append(self.linear_list[b](input[..., in_features_p1:in_features_p2]))
                else:
                    if b == 0:
                        outs.append(self.linear_list[0](input[..., :self.in_features_list[0]]))
                        continue
                    old_features = sum(self.in_features_list[:b])
                    curr_features = sum(self.in_features_list[:b+1])

                    if self.self_attn:
                        out = 0
                    else:
                        out = self.linear_list[b](input[..., old_features:curr_features])
                    if self.fix_attn:
                        proj_input = input[..., :curr_features] if self.self_attn else input[..., :old_features]
                        proj = self.proj_blocks[b](proj_input)
                        # if input.shape[-1] == self.head_dim * 14 and debug:
                        #    print("Display fixed attention for block {}".format(b))
                        #    print(self.proj_blocks[b].attn[0, :3])
                    else:
                        attn = self.attn_blocks[b](input[..., :old_features], input[..., old_features:curr_features])
                        proj_input = input[..., :curr_features] if self.self_attn else input[..., :old_features]
                        proj = self.proj_blocks[b](proj_input, attn)
                        # if input.shape[-1] == self.head_dim * 14:
                        #     print("DEBUG ATTENTION! for block {}".format(b))
                        #     torch.manual_seed(0)
                        #     debug_input1 = torch.randn(input[..., :old_features].shape).cuda()
                        #     debug_input2 = torch.randn(input[..., old_features:curr_features].shape).cuda()
                        #     print(debug_input1[0, 0, :10])
                        #     print(debug_input2[0, 0, :10])
                        #     debug_attn = self.attn_blocks[b](debug_input1, debug_input2)
                        #     print(debug_attn[0, 0])
                        #     pause()
                        #
                        # if input.shape[-1] == self.head_dim * 14 and debug:
                        #     print("Display non-fixed attention for block {}".format(b))
                        #     print(attn[0, :3])
                        #     print("record mean is {}".format(self.attn_blocks[b].mean_attn))
                    if len(proj.shape) > len(input.shape):
                        proj = proj.squeeze(1)
                    if self.self_attn:
                        out = proj
                    else:
                        out += proj
                    outs.append(out)
        if self.simple_proj or self.block_length == 1:
            if self.stack or not self.split:
                outs.append(self.linear_list[-1](input))
            else:
                outs.append(self.linear_list[-1](input[..., sum(self.in_features_list[:-1]):]))
        else:
            old_features = sum(self.in_features_list[:-1])
            if self.self_attn:
                out = 0
            else:
                out = self.linear_list[-1](input[..., old_features:])
            attn = self.curr_attn_block(input[..., :old_features], input[..., old_features:])
            proj_input = input if self.self_attn else input[..., :old_features]
            proj = self.proj_blocks[-1](proj_input, attn)
            # if attn.shape[-1] == 14 and debug:
            #     print("Display current attention")
            #     print(attn[0, :3])
            #     pause()
            if len(proj.shape) > len(input.shape):
                proj = proj.squeeze(1)
            if self.self_attn:
                out = proj
            else:
                out += proj
            outs.append(out)
        return torch.cat(outs, dim=-1)

def pause():
    c = input()

class split_Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        simple_proj=False,
        split=True,
        stack=True,
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

        self.split = split
        self.stack = stack
        self.simple_proj = simple_proj

        assert self.stack, "stack=False not implemented"
        assert self.simple_proj, "simple_proj=False not implemented"


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

    def expand(self, in_channels, out_channels, fix_old = True, init_new = True):
        if not self.split:
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