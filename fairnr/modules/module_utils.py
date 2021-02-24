# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from fairseq.modules import LayerNorm
from fairseq.utils import get_activation_fn


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    return m
    

class PosEmbLinear(nn.Module):

    def __init__(self, in_dim, out_dim, no_linear=False, scale=1, *args, **kwargs):
        super().__init__()
        assert out_dim % (2 * in_dim) == 0, "dimension must be dividable"
        half_dim = out_dim // 2 // in_dim
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        
        self.emb = nn.Parameter(emb, requires_grad=False)
        self.linear = Linear(out_dim, out_dim) if not no_linear else None
        self.scale = scale
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cat_input = False

    def forward(self, x):
        assert x.size(-1) == self.in_dim, "size must match"
        sizes = x.size()
        x = self.scale * x.unsqueeze(-1) @ self.emb.unsqueeze(0)
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        x = x.view(*sizes[:-1], self.out_dim)
        if self.linear is not None:
            return self.linear(x)
        return x


class NeRFPosEmbLinear(nn.Module):

    def __init__(self, in_dim, out_dim, angular=False, no_linear=False, cat_input=False):
        super().__init__()
        assert out_dim % (2 * in_dim) == 0, "dimension must be dividable"
        L = out_dim // 2 // in_dim
        emb = torch.exp(torch.arange(L, dtype=torch.float) * math.log(2.))
        if not angular:
            emb = emb * math.pi

        self.emb = nn.Parameter(emb, requires_grad=False)
        self.angular = angular
        self.linear = Linear(out_dim, out_dim) if not no_linear else None
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cat_input = cat_input

    def forward(self, x):
        assert x.size(-1) == self.in_dim, "size must match"
        sizes = x.size() 
        inputs = x.clone()

        if self.angular:
            x = torch.acos(x.clamp(-1 + 1e-6, 1 - 1e-6))
        x = x.unsqueeze(-1) @ self.emb.unsqueeze(0)
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        x = x.view(*sizes[:-1], self.out_dim)
        if self.linear is not None:
            x = self.linear(x)
        if self.cat_input:
            x = torch.cat([x, inputs], -1)
        return x

    def extra_repr(self) -> str:
        outstr = 'Sinusoidal (in={}, out={}, angular={})'.format(
            self.in_dim, self.out_dim, self.angular)
        if self.cat_input:
            outstr = 'Cat({}, {})'.format(outstr, self.in_dim)
        return outstr


class GroupDense(nn.Module):
    def __init__(self, in_dim, out_dim, groups=2, bias=True):
        super().__init__()
        self.groups = groups
        self.in_dim = in_dim
        self.out_dim = out_dim

        assert (in_dim % groups == 0) and (out_dim % groups == 0)
        self.ch_per_in_group = in_dim // groups
        self.ch_per_out_group = out_dim // groups

        # TODO: Remove rand initialization
        self.weight = Parameter(torch.rand(groups, self.ch_per_in_group, self.ch_per_out_group))
        if bias:
            self.bias = Parameter(torch.rand(out_dim))
        else:
            self.register_parameter('bias', None)

        # TODO: Add proper weight initialization
        # self.reset_parameters()

    # def reset_parameters(self):
    #     init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    #     if self.bias is not None:
    #         fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
    #         bound = 1 / math.sqrt(fan_in)
    #         init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        # Reshape for batch matmul
        N = x.size(0)
        x = x.view(N, self.groups, self.ch_per_in_group)

        # Perform fused op
        output = torch.einsum('bgi,gij->bgj', x, self.weight) # NOTE: Only einsum implementation seems to have optimized backprop memory consumption

        # Reshape back to [N, C_out]
        output = output.reshape(N, self.out_dim)

        # Add bias
        if self.bias is not None:
            output += self.bias

        return output


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels = x.data.size()
        assert num_channels % self.groups == 0
        group_channels = num_channels // self.groups
        
        x = x.reshape(batchsize, group_channels, self.groups)
        x = x.permute(0, 2, 1)
        x = x.reshape(batchsize, num_channels)

        return x

class ShuffleNetLayer(nn.Module):
    def __init__(self, in_dim, out_dim, with_ln=True, groups=2, with_act=True):
        super().__init__()
        self.net = [GroupDense(in_dim, out_dim, groups=groups)]
        if with_ln:
            self.net += [nn.LayerNorm([out_dim])]
        if with_act:
            self.net += [nn.ReLU()]
        self.net += [ChannelShuffle(groups)]
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


class FCLayer(nn.Module):
    """
    Reference:
        https://github.com/vsitzmann/pytorch_prototyping/blob/10f49b1e7df38a58fd78451eac91d7ac1a21df64/pytorch_prototyping.py
    """
    def __init__(self, in_dim, out_dim, with_ln=True):
        super().__init__()
        self.net = [nn.Linear(in_dim, out_dim)]
        if with_ln:
            self.net += [nn.LayerNorm([out_dim])]
        self.net += [nn.ReLU()]
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x) 


class FCBlock(nn.Module):
    def __init__(self,
                 hidden_ch,
                 num_hidden_layers,
                 in_features,
                 out_features,
                 outermost_linear=False,
                 with_ln=True):
        super().__init__()

        self.net = []
        self.net.append(FCLayer(in_features, hidden_ch, with_ln))
        for i in range(num_hidden_layers):
            self.net.append(FCLayer(hidden_ch, hidden_ch, with_ln))
        if outermost_linear:
            self.net.append(Linear(hidden_ch, out_features))
        else:
            self.net.append(FCLayer(hidden_ch, out_features, with_ln))
        self.net = nn.Sequential(*self.net)
        self.net.apply(self.init_weights)

    def __getitem__(self, item):
        return self.net[item]

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')

    def forward(self, input):
        return self.net(input)


class InvertableMapping(nn.Module):
    def __init__(self, style='simple'):
        super().__init__()
        self.style = style

    def f(self, x):  # (0, 1) --> (0, +inf)
        if self.style == 'simple':
            return x / (1 - x + 1e-7)
        raise NotImplementedError
    
    def g(self, y):  # (0, +inf) --> (0, 1)
        if self.style == 'simple':
            return y / (1 + y)
        raise NotImplementedError

    def dy(self, x):
        if self.style == 'simple':
            return 1 / ((1 - x) ** 2 + 1e-7)
        raise NotImplementedError