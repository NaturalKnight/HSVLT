# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2022-3-22
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2022 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import math
import torch
import torch.nn as nn
import numpy as np

from collections import OrderedDict

from .timm_models.vision_transformer import *
from .timm_models.util.layers import DropPath
from .timm_models.util.layers import Mlp as Mlp3
from .factory import register_model

from torch.nn.modules.utils import _pair as to_2tuple

import torch.utils.checkpoint as checkpoint
from timm.models.layers import trunc_normal_, DropPath
import torch.nn.functional as F

# from .visual_feature import get_feature, visualize_feature_map_sum

# all_dict = {}




class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.lang = nn.Linear(dim*2, dim, bias=qkv_bias)
        self.query = nn.Linear(dim, dim, bias=qkv_bias)
        self.key = nn.Linear(dim, dim, bias=qkv_bias)
        self.value = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, vfeat, tfeat):
        # print(vfeat.shape, tfeat.shape)
        # B, H, W, C = vfeat.shape
        B, C, H, W = vfeat.shape
        Nv = H*W
        Nt = tfeat.size(1)
        # vfeat = vfeat.reshape(B, Nv, C)
        vfeat = vfeat.permute(0, 3, 1, 2).contiguous().reshape(B, Nv, C)
        # tfeat = self.lang(tfeat)
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        # print(vfeat.shape, tfeat.shape, self.num_heads)
        q = self.query(tfeat).reshape(B, Nt, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.key(vfeat).reshape(B, Nv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.value(vfeat).reshape(B, Nv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, Nt, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.lang = nn.Linear(dim*2, dim, bias=qkv_bias)
        self.query = nn.Linear(dim, dim, bias=qkv_bias)
        self.key = nn.Linear(dim, dim, bias=qkv_bias)
        self.value = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.vis_project = nn.Sequential(nn.Conv1d(dim, dim, 1, 1),  # the init function sets bias to 0 if bias is True
                                         nn.GELU(),
                                         nn.Dropout(proj_drop)
                                        )
        self.project_mm = nn.Sequential(nn.Conv1d(dim, dim, 1, 1),
                                        nn.GELU(),
                                        nn.Dropout(proj_drop)
                                        )

    def forward(self, vfeat, tfeat):
        # print(vfeat.shape, tfeat.shape)
        # B, H, W, C = vfeat.shape
        B, C, H, W = vfeat.shape
        Nv = H*W
        Nt = tfeat.size(1)
        # vfeat = vfeat.reshape(B, Nv, C)
        vfeat = vfeat.permute(0, 2, 3, 1).reshape(B, Nv, C)
        tf = self.vis_project(tfeat.permute(0, 2, 1))
        # tfeat = self.lang(tfeat)
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        q = self.query(tfeat).reshape(B, Nt, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.key(vfeat).reshape(B, Nv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.value(vfeat).reshape(B, Nv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, Nt, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        mm = torch.mul(tf, x.permute(0, 2, 1))
        mm = self.project_mm(mm)  # (B, dim, H*W)

        mm = mm.permute(0, 2, 1)  # (B, H*W, dim)

        return mm, attn


class TextHead(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
                    
    def forward(self, x):
        x = torch.sum(x * self.weight, 2)
        if self.bias is not None:
            x = x + self.bias
        return x



class InteractionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp3(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)
        self.res_gate = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),
            nn.Tanh()
        )
        # nn.init.zeros_(self.res_gate[0].weight)
        # nn.init.zeros_(self.res_gate[2].weight)

    def forward(self, ints):
        x, tfeat, tfeat_re, attn = ints
        # print(x.shape, tfeat.shape)
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)

        tfeat_re, attn = self.attn(x, tfeat)
        tfeat = tfeat + self.drop_path(tfeat_re)
        # tfeat = tfeat + self.drop_path(self.mlp(self.norm2(tfeat)))
        tfeat = tfeat + self.res_gate(tfeat) * tfeat

        outs = []
        outs.append(x)
        outs.append(tfeat)
        outs.append(tfeat_re)
        outs.append(attn)
        # outs.append(attn)
        return outs



class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        
        # print(dim, mlp_ratio)
        mlp_ratio = int(mlp_ratio)
        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape

        
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)

        return x

class ConvMod(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a = nn.Sequential(
                nn.Conv2d(dim, dim, 1),
                nn.GELU(),
                nn.Conv2d(dim, dim, 11, padding=5, groups=dim)
        )

        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.attn = Attention2(dim)

    def forward(self, x, tfeat):
        B, C, H, W = x.shape
        
        x = self.norm(x)   
        a = self.a(x)
        m, t, attn = self.attn(x, tfeat, H, W)

        x = (a + m) * self.v(x)
        x = self.proj(x)

        return x, tfeat




class SimpleDecoding(nn.Module):
    def __init__(self, c4_dims, factor=2):
        super(SimpleDecoding, self).__init__()
        # embed_dims=[96, 192, 384, 768]
        embed_dims=[128, 256, 512, 1024]
        c4_dims = embed_dims[3]
        hidden_size = c4_dims//factor
        c4_size = embed_dims[3]
        c3_size = embed_dims[2]
        c2_size = embed_dims[1]
        c1_size = embed_dims[0]
        # v1_size = 64
        # self.input_shape = 480
        # norm_cfg=dict(type='BN', requires_grad=True)

        self.conv1_4 = nn.Conv1d(c4_size+c3_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_4 = nn.LayerNorm(hidden_size)
        self.relu1_4 = nn.ReLU()
        self.conv2_4 = nn.Conv1d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_4 = nn.LayerNorm(hidden_size)
        self.relu2_4 = nn.ReLU()

        self.conv1_3 = nn.Conv1d(hidden_size + c2_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_3 = nn.LayerNorm(hidden_size)
        self.relu1_3 = nn.ReLU()
        self.conv2_3 = nn.Conv1d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_3 = nn.LayerNorm(hidden_size)
        self.relu2_3 = nn.ReLU()

        self.conv1_2 = nn.Conv1d(hidden_size + c1_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_2 = nn.LayerNorm(hidden_size)
        self.relu1_2 = nn.ReLU()
        self.conv2_2 = nn.Conv1d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_2 = nn.LayerNorm(hidden_size)
        self.relu2_2 = nn.ReLU()

        self.conv1 = nn.Conv1d(hidden_size, 2, 1)
        # self.segmentation_head = nn.Conv2d(hidden_size//4, 2, 3, padding=1)

    def forward(self, x_c4, x_c3, x_c2, x_c1):
        # fuse Y4 and Y3
        # print(x_c4.shape, x_c3.shape)
        x = torch.cat([x_c4, x_c3], dim=1)
        # print(x.shape)
        x = self.conv1_4(x)
        x = self.bn1_4(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.relu1_4(x)
        x = self.conv2_4(x)
        x = self.bn2_4(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.relu2_4(x)
        
        # fuse top-down features and Y2 features
        # print(x.shape, x_c2.shape)
        x = torch.cat([x, x_c2], dim=1)
        x = self.conv1_3(x)
        x = self.bn1_3(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.relu1_3(x)
        x = self.conv2_3(x)
        x = self.bn2_3(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.relu2_3(x)
        
        x = torch.cat([x, x_c1], dim=1)
        x = self.conv1_2(x)
        x = self.bn1_2(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.relu1_2(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.relu2_2(x)
        
        return x.permute(0, 2, 1)

class HSVLT(nn.Module):
    # def __init__(self, in_chans=3, num_classes=1000, 
    #              depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
    #              layer_scale_init_value=1e-6, head_init_scale=1.,
    #              ):
    def __init__(self, pretrained, cfg, embed_dim=768, depth=12, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        
        if cfg.embed_type == 'bert' or cfg.embed_type == 'glove':
            self.text_feats = torch.tensor(np.load(cfg.embed_path), dtype=torch.float32).cuda()
        elif cfg.embed_type == 'random':
            self.text_feats = torch.eye(cfg.num_classes).cuda()

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        self.langs = nn.ModuleList()
        for i in range(4):
            if i == 0:
                lang = nn.Linear(768, dims[i])
            else:
                lang = nn.Linear(dims[i-1], dims[i])
            self.langs.append(lang)
        for i in range(4):
            stage = nn.Sequential(
                *[InteractionBlock(dim=dims[i], num_heads=cfg.num_heads, drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
                # *[Block5(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)
        self.cfg = cfg
        self.attn = Attention(dims[3], num_heads=cfg.num_heads)
        # self.text_head = TextHead(dims[3], cfg.num_classes)
        self.text_head2 = TextHead(dims[0]+dims[1]+dims[2]+dims[3], cfg.num_classes)
        # self.text_head4 = TextHead(dims[3]//2, cfg.num_classes)
        # self.text_head5 = TextHead(512, cfg.num_classes)
        self.fuse = SimpleDecoding(dims[3])
        # self.fuse = LightHamHead(
        #         ham_channels=512,
        #         # ham_channels=1024,
        #         dropout_ratio=0.1,
        #         # ham_kwargs=dict(MD_R=16),
        #         ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True))
        # self.text_head3 = TextHead((dims[0]+dims[1]+dims[2]+dims[3])*2, cfg.num_classes)
        
        # self.apply(self.init_weights)
        # self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        
        pretrained="./checkpoints/convnext_small_22k_1k_384.pth"
        
        print(pretrained)

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)


        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            ckpt_path = pretrained
            ckpt = _load_checkpoint(
                ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)

        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        batch_size = x.size(0)
        tfeat = torch.stack([self.text_feats for _ in range(batch_size)], dim=0)
        if self.cfg.embed_type == 'random':
            tfeat = self.text_linear(tfeat)
        
        # x = self.forward_features(x)
        vfeat = x
        t_end= []
        v_end= []
        attn = []
        for i in range(4):
            vfeat = self.downsample_layers[i](vfeat)
            tfeat = self.langs[i](tfeat)
            ints = []
            tfeat_re = []
            ints.append(vfeat)
            ints.append(tfeat)
            ints.append(tfeat_re)
            ints.append(attn)
            outs = self.stages[i](ints)
            vfeat, tfeat, tfeat_re, attn = outs
            v_end.append(vfeat)
            t_end.append(tfeat)
            t_end.append(tfeat_re)
        
        # t, attn = self.attn(vfeat, tfeat)

        # t1, t2, t3, t4 = t_end
        # t = torch.cat([t1, t2, t3, t4], dim=2)
        # logits = self.text_head2(t)

        # visual
        # v1, v2, v3, v4 = v_end
        # all_dict["x1"] = v1
        # all_dict["x2"] = v2
        # all_dict["x3"] = v3
        # all_dict["x4"] = v4
        # get_feature(all_dict)

        t1, re1, t2, re2, t3, re3, t4, re4 = t_end
        # t = torch.cat([t1, t2, t3, t4], dim=2)
        t = torch.cat([re1, re2, re3, re4], dim=2)
        # t = self.fuse(re4.permute(0, 2, 1), re3.permute(0, 2, 1), re2.permute(0, 2, 1), re1.permute(0, 2, 1))
        # t = self.fuse(t4.permute(0, 2, 1), t3.permute(0, 2, 1), t2.permute(0, 2, 1), t1.permute(0, 2, 1))
        # print(t.shape)
        # t = torch.cat([t1, re1, t2, re2, t3, re3, t4, re4], dim=2)
        logits = self.text_head2(t)
        # logits = self.text_head3(t)
        # logits = self.text_head4(t)


        
        # logits = self.text_head(tfeat)
        
        if self.training:
            return logits
        
        return logits, attn[..., 1:]
        # return x

class _MatrixDecomposition2DBase(nn.Module):
    def __init__(self, args=dict()):
        super().__init__()

        self.spatial = args.setdefault('SPATIAL', True)

        self.S = args.setdefault('MD_S', 1)
        self.D = args.setdefault('MD_D', 512)
        self.R = args.setdefault('MD_R', 64)

        self.train_steps = args.setdefault('TRAIN_STEPS', 6) # 3
        self.eval_steps = args.setdefault('EVAL_STEPS', 7) # 4

        self.inv_t = args.setdefault('INV_T', 100)
        self.eta = args.setdefault('ETA', 0.9)

        self.rand_init = args.setdefault('RAND_INIT', True)
        self.f_l1 = nn.Sequential(
            nn.Conv1d(768, self.D, kernel_size=1, stride=1),
        )

    def _build_bases(self, B, S, D, R, cuda=False):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    # @torch.no_grad()
    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):
        # B, C, H, W = x.shape
        B, C, N2 = x.shape

        # (B, C, N) -> (B * S, D, N)
        if self.spatial:
            D = C // self.S
            N = N2
            x = x.view(B * self.S, D, N)
        else:
            D = N2
            N = C // self.S
            x = x.view(B * self.S, N, D).transpose(1, 2)

        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, D, self.R, cuda=True)
            self.register_buffer('bases', bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, cuda=True)
        else:
            bases = self.bases.repeat(B, 1, 1)
            

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        # (B * S, D, N) -> (B, C, H, W)
        if self.spatial:
            x = x.view(B, C, N2)
        else:
            x = x.transpose(1, 2).view(B, C, N2)

        # (B * H, D, R) -> (B, H, N, D)
        bases = bases.view(B, self.S, D, self.R)

        return x


class NMF2D(_MatrixDecomposition2DBase):
    def __init__(self, args=dict()):
        super().__init__(args)

        self.inv_t = 1

    def _build_bases(self, B, S, D, R, cuda=False):
        if cuda:
            bases = torch.rand((B * S, D, R)).cuda()
        else:
            bases = torch.rand((B * S, D, R))

        bases = F.normalize(bases, dim=1)

        return bases

    # @torch.no_grad()
    def local_step(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)

        return coef


class Hamburger(nn.Module):
    def __init__(self,
                 ham_channels=512,
                 ham_kwargs=dict(),
                 norm_cfg=None):
        super().__init__()

        self.ham_in = nn.Conv1d(ham_channels, ham_channels, 3, padding=1, bias=False)

        self.ham = NMF2D(ham_kwargs)

        self.ham_out = nn.Conv1d(ham_channels, ham_channels, 3, padding=1, bias=False)

    def forward(self, x):
        enjoy = self.ham_in(x)
        enjoy = F.relu(enjoy, inplace=True)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        ham = F.relu(x + enjoy, inplace=True)

        return ham

class LightHamHead(nn.Module):
    def __init__(self, ham_norm_cfg, dropout_ratio=0., ham_channels=512, factor=2, ham_kwargs=dict()):
        super(LightHamHead, self).__init__()
        # embed_dims=[96, 192, 384, 768]
        embed_dims=[128, 256, 512, 1024]
        c4_dims = embed_dims[3]
        hidden_size = c4_dims//factor
        c4_size = embed_dims[3]
        c3_size = embed_dims[2]
        c2_size = embed_dims[1]
        c1_size = embed_dims[0]
        # v1_size = 64
        # self.input_shape = 480
        self.ham_channels = ham_channels
        
        # self.in_channels = embed_dims[1:]
        self.in_channels = embed_dims[:]
        self.align_corners=False
        self.squeeze = nn.Conv1d(sum(self.in_channels), self.ham_channels, 3, padding=1, bias=False)
        self.align = nn.Conv1d(self.ham_channels, hidden_size, 3, padding=1, bias=False)

        self.hamburger = Hamburger(self.ham_channels, ham_kwargs)

    def forward(self, x_c4, x_c3, x_c2, x_c1):
        
        x = torch.cat([x_c4, x_c3, x_c2, x_c1], dim=1)
        # x = torch.cat([x4, x3, x2], dim=1)
        # print(x.shape, x4.shape, x3.shape, x2.shape)
        x = self.squeeze(x)
        x = self.hamburger(x)

        # align
        x = self.align(x)

        return x.permute(0, 2, 1)



class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x




@register_model
def hsvlt(pretrained="./checkpoints/convnext_small_22k_1k_384.pth",in_22k=False, **kwargs):
    model = HSVLT(pretrained="./checkpoints/convnext_small_22k_1k_384.pth", depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    return model
