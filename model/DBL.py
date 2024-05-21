import torch.nn as nn
import torch
import torch.nn.functional as F
from model.convlstm import ConvLSTM
from torch import Tensor
from timm.models.layers import DropPath
from typing import List
from functools import partial

class DBL(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        inconv=[32, 64],
        n_levels=5,
        n_channels=64,
        hidden_size=88,
        input_shape=(128, 128),
        mid_conv=True,
        pad_value=0,
    ):

        super(DBL, self).__init__()
        self.pad_value = pad_value
        self.patch_size = 2
        self.patch_stride = 2
        self.patch_size2 = 2  # for subsequent layers
        self.patch_stride2 = 2
        self.num_stages = 3
        self.embed_dim = 64
        self.n_div = 4
        self.width = 128
        self.patch_norm = True
        self.mlp_ratio = 2
        self.depths = (1,2)
        self.In_dim = [64,64]
        self.Out_dim = [64,128]
        self.drop_path_rate = 0.1
        self.norm_layer = nn.GroupNorm
        self.act_layer = partial(nn.ReLU, inplace=True)
        self.inconv = ConvBlock(
            nkernels=[input_dim] + inconv, norm="group", pad_value=pad_value
        )
        self.aspp = ASPP(4*self.width, [6, 12, 18])
        self.TA = TemporalAttention(12, ratio=4)
        self.ca1 = ChannelAttention(64)
        self.sa1 = SpatialAttention()
        if mid_conv:
            dim = 128
            self.mid_conv = ConvBlock(
                nkernels=[64+ 64 , dim],
                pad_value=pad_value,
                norm="group",
            )
        # self.convlstm = ConvLSTM(
        #     input_dim=32,
        #     input_size=input_shape,
        #     hidden_dim=hidden_size,
        #     kernel_size=(3, 3),
        #     return_all_layers=False,
        # )
        # self.convlstm2 = ConvLSTM(
        #     input_dim=4*self.width,
        #     input_size=input_shape,
        #     hidden_dim=hidden_size,
        #     kernel_size=(3, 3),
        #     return_all_layers=False,
        # )
        self.convlstmcan = ConvLSTM(
            input_dim=dim,
            input_size=input_shape,
            hidden_dim=hidden_size,
            kernel_size=(3, 3),
            return_all_layers=False,
        )
        self.outconv = nn.Conv2d(
            in_channels=hidden_size, out_channels=num_classes, kernel_size=1
        )
        #FasterNet defination
        self.dpr = [x.item()
                    for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]
        #1
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            in_chans=inconv[-1],
            embed_dim=self.embed_dim,
            norm_layer=self.norm_layer if self.patch_norm else None
        )

        self.stage = nn.ModuleList(
            BasicStage(
                dim=int(self.Out_dim[i]),
                n_div=self.n_div,
                depth=self.depths[i],
                mlp_ratio=self.mlp_ratio,
                drop_path=self.dpr[sum(self.depths[:i]):sum(self.depths[:i + 1])],
                norm_layer=self.norm_layer,
                act_layer=self.act_layer,
            )
            for i in range(self.num_stages - 1)
        )

        self.redu = nn.ModuleList(
            PatchMerging(
                patch_size2=self.patch_size2,
                patch_stride2=self.patch_stride2,
                In_dim=self.In_dim[i],
                Out_dim=self.Out_dim[-1],
                norm_layer=self.norm_layer,
            )
            for i in range(self.num_stages - 2)
        )
        #JPU defination
        self.jpu = JPU(in_channels = [self.In_dim[0]]+self.Out_dim, width=self.width, norm_layer=nn.GroupNorm, up_kwargs=None)
    def forward(self, input, batch_positions=None):
        pad_mask = (
            (input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
        )  # BxT pad mask
        pad_mask = pad_mask if pad_mask.any() else None

        out = self.inconv.smart_forward(input)
        branch1 = out
        branch2 = out
        feature_maps = [branch2]
        branch2 = self.patch_embed.smart_forward(branch2)
        for i in range(self.num_stages - 1):
            branch2 = self.stage[i].smart_forward(branch2)
            feature_maps.append(branch2)
            if i < self.num_stages - 2:
                branch2 = self.redu[i].smart_forward(branch2)
        branch2 = self.jpu(feature_maps[-3],feature_maps[-2],feature_maps[-1])
        branch2 = self.aspp.smart_forward(branch2)
        # branch1 = self.aspp.smart_forward(branch1)
        branchT = branch1.permute(0,2,1,3,4)
        B, C, T, H, W = branchT.shape
        branchT = branchT.reshape(B * C, T, H, W)
        branchT = self.TA(branchT) * branchT
        branchT = branchT.reshape(B, C, T, H, W).permute(0, 2, 1, 3, 4)

        B, T, C, H, W = branch1.shape
        branch1 = self.ca1.smart_forward(branch1).reshape(B * T, C, 1, 1) * branch1.reshape(B * T, C, H, W)
        branch1 = branch1.reshape(B, T, C, H, W)
        branch1 = branch1 + branchT
        branch1 = self.sa1.smart_forward(branch1).reshape(B * T, 1, H, W) * branch1.reshape(B * T, C, H, W)
        branch1 = branch1.reshape(B, T, C, H, W)
        out = torch.cat((branch1, branch2), dim=2)
        out = self.mid_conv.smart_forward(out)
        _, out = self.convlstmcan(out, pad_mask=pad_mask)
        out = out[0][1]
        out = self.outconv(out)
        return out

        # _, branch2 = self.convlstm2(branch2, pad_mask=pad_mask)
        # branch2 =branch2[0][1]
        # pred2 = self.outconv(branch2)

        # return branch2


        # out = self.aspp.smart_forward(out)
        # B, T, C, H, W = out.shape
        # out = self.ca1.smart_forward(out).reshape(B * T, C, 1, 1) * out.reshape(B * T, C, H, W)
        # out = out.reshape(B, T, C, H, W)
        # out = self.sa1.smart_forward(out).reshape(B * T, 1, H, W) * out.reshape(B * T, C, H, W)
        # out = out.reshape(B, T, C, H, W)
        # _, branch1 = self.convlstm(branch1, pad_mask=pad_mask)
        # branch1 = branch1[0][1]
        # pred1 = self.outconv(branch1)
        # out = 0.7*pred1 +0.3*pred2
        #
        # return out

class TemporallySharedBlock(nn.Module):
    """
    Helper module for convolutional encoding blocks that are shared across a sequence.
    This module adds the self.smart_forward() method the the block.
    smart_forward will combine the batch and temporal dimension of an input tensor
    if it is 5-D and apply the shared convolutions to all the (batch x temp) positions.
    """
#这个类用于将批量维度b和时间维度t打包为一个维度
    def __init__(self, pad_value=None):
        super(TemporallySharedBlock, self).__init__()
        self.out_shape = None
        self.pad_value = pad_value

    def smart_forward(self, input):
        if len(input.shape) == 4:
            return self.forward(input)
        else:
            b, t, c, h, w = input.shape


            out = input.contiguous().view(b * t, c, h, w)

            out = self.forward(out)
            _, c, h, w = out.shape
            out = out.view(b, t, c, h, w)
            return out
#处理时将b，t维度合并，返回时仍返回b,t,c,h,w五个维度的tensor

class PatchEmbed(TemporallySharedBlock):

    def __init__(self, patch_size, patch_stride, in_chans, embed_dim, norm_layer):
        super(PatchEmbed, self).__init__(pad_value=None)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, bias=False)
        self.norm = norm_layer(16,embed_dim)
        # if norm_layer == 'BN':
        #     norm_layer = nn.BatchNorm2d
        # if norm_layer is not None:
        #     self.norm = norm_layer(embed_dim)
        # else:
        #     self.norm = nn.Identity()


    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.proj(x))
        return x
class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x
class MLPBlock(nn.Module):

    def __init__(self,
                 dim,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 act_layer,
                 norm_layer,
                 ):

        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(16,mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div
        )


    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

class BasicStage(TemporallySharedBlock):

    def __init__(self,
                 dim,
                 depth,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 norm_layer,
                 act_layer
                 ):
        super(BasicStage, self).__init__(pad_value=None)


        blocks_list = [
            MLPBlock(
                dim=dim,
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        return x

class PatchMerging(TemporallySharedBlock):

    def __init__(self, patch_size2, patch_stride2, In_dim, Out_dim, norm_layer):
        super(PatchMerging, self).__init__(pad_value=None)
        self.reduction = nn.Conv2d(In_dim, Out_dim, kernel_size=patch_size2, stride=patch_stride2, bias=False)

        self.norm = norm_layer(16,Out_dim)


    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.reduction(x))
        return x

class ConvLayer(nn.Module):
    def __init__(
            self,
            nkernels,
            norm="batch",
            k=3,
            s=1,
            p=1,
            n_groups=4,
            last_relu=True,
            padding_mode="reflect",
    ):
        super(ConvLayer, self).__init__()
        layers = []
        if norm == "batch":
            nl = nn.BatchNorm2d
        elif norm == "instance":
            nl = nn.InstanceNorm2d
        elif norm == "group":
            nl = lambda num_feats: nn.GroupNorm(
                num_channels=num_feats,
                num_groups=n_groups,
            )
        else:
            nl = None
        for i in range(len(nkernels) - 1):
            # if nkernels=[10,64,64], then len=3
            layers.append(
                nn.Conv2d(
                    in_channels=nkernels[i],
                    out_channels=nkernels[i + 1],
                    kernel_size=k,
                    padding=p,
                    stride=s,
                    padding_mode=padding_mode,
                )
            )
            if nl is not None:
                layers.append(nl(nkernels[i + 1]))
            # append用于在列表末尾添加对象
            if last_relu:
                layers.append(nn.ReLU())
            elif i < len(nkernels) - 2:
                layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, input):
        return self.conv(input)

class ConvBlock(TemporallySharedBlock):
    def __init__(
            self,
            nkernels,
            pad_value=None,
            norm="batch",
            last_relu=True,
            padding_mode="reflect",
    ):
        super(ConvBlock, self).__init__(pad_value=pad_value)
        self.conv = ConvLayer(
            nkernels=nkernels,
            norm=norm,
            last_relu=last_relu,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        return self.conv(input)
class TemporalAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(TemporalAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class ChannelAttention(TemporallySharedBlock):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(TemporallySharedBlock):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(16,out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(16,out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(TemporallySharedBlock):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 64
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(16, out_channels),
            nn.ReLU()))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.GroupNorm(16, out_channels),
            nn.ReLU(),
            nn.Dropout(0.1))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = norm_layer(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class JPU(nn.Module):
    def __init__(self, in_channels, width=512, norm_layer=None, up_kwargs=None):
        super(JPU, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            norm_layer(16,width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            norm_layer(16,width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            norm_layer(16,width),
            nn.ReLU(inplace=True))

        self.dilation1 = nn.Sequential(
            SeparableConv2d(3 * width, width, kernel_size=3, padding=1, dilation=1, bias=False),
            norm_layer(16,width),
            nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(
            SeparableConv2d(3 * width, width, kernel_size=3, padding=2, dilation=2, bias=False),
            norm_layer(16,width),
            nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(
            SeparableConv2d(3 * width, width, kernel_size=3, padding=4, dilation=4, bias=False),
            norm_layer(16,width),
            nn.ReLU(inplace=True))
        self.dilation4 = nn.Sequential(
            SeparableConv2d(3 * width, width, kernel_size=3, padding=8, dilation=8, bias=False),
            norm_layer(16,width),
            nn.ReLU(inplace=True))

    def forward(self, *inputs):
        b,t,c,h,w = inputs[-3].shape
        F3 = inputs[-3].view(b * t, c, h, w)
        b, t, c, h, w = inputs[-2].shape
        F2 = inputs[-2].view(b * t, c, h, w)
        b, t, c, h, w = inputs[-1].shape
        F1 = inputs[-1].view(b * t, c, h, w)
        feats = [self.conv5(F1), self.conv4(F2), self.conv3(F3)]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w), mode='bicubic', align_corners=False)
        feats[-3] = F.interpolate(feats[-3], (h, w), mode='bicubic', align_corners=False)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat).view(b,t,-1,h,w), self.dilation2(feat).view(b,t,-1,h,w), self.dilation3(feat).view(b,t,-1,h,w), self.dilation4(feat).view(b,t,-1,h,w)],
                         dim=2)

        return  feat