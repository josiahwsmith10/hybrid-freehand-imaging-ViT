# MODEL
import torch
import torch.nn as nn
import numpy as np

#THIS IS THE ADDED LIBRARY FROM THE GITHUB LUCIDRAINS MOBILEVIT
from einops import rearrange
from einops.layers.torch import Reduce


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

# change dimension and fn to take inputs from the list of arguments, **kwargs may need to be changed
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# change additional init inputs to arg
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# dim from args
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)

# input from args
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# inputs from arg
class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            out = out + x
        return out


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d',
                      ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)',
                      h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


#take inputs from arg
class MobileViT(nn.Module):
    def __init__(
        self,
        image_size,
        dims,
        channels,
        num_classes,
        expansion=4,
        kernel_size=3,
        patch_size=(2, 2),
        depths=(2, 4, 3)
    ):
        super().__init__()
        assert len(dims) == 3, 'dims must be a tuple of 3'
        assert len(depths) == 3, 'depths must be a tuple of 3'

        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        init_dim, *_, last_dim = channels

        self.conv1 = conv_nxn_bn(3, init_dim, stride=2)

        self.stem = nn.ModuleList([])
        self.stem.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.stem.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.stem.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.stem.append(MV2Block(channels[2], channels[3], 1, expansion))

        self.trunk = nn.ModuleList([])
        self.trunk.append(nn.ModuleList([
            MV2Block(channels[3], channels[4], 2, expansion),
            MobileViTBlock(dims[0], depths[0], channels[5],
                           kernel_size, patch_size, int(dims[0] * 2))
        ]))

        self.trunk.append(nn.ModuleList([
            MV2Block(channels[5], channels[6], 2, expansion),
            MobileViTBlock(dims[1], depths[1], channels[7],
                           kernel_size, patch_size, int(dims[1] * 4))
        ]))

        self.trunk.append(nn.ModuleList([
            MV2Block(channels[7], channels[8], 2, expansion),
            MobileViTBlock(dims[2], depths[2], channels[9],
                           kernel_size, patch_size, int(dims[2] * 4))
        ]))

        self.to_logits = nn.Sequential(
            conv_1x1_bn(channels[-2], last_dim),
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(channels[-1], num_classes, bias=False)
        )

    def forward(self, x):
        x = self.conv1(x)

        for conv in self.stem:
            x = conv(x)

        for conv, attn in self.trunk:
            x = conv(x)
            x = attn(x)

        return self.to_logits(x)


# CLASS DEFINITION WITH ARGS INPUT
class HFFH_ViT(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.image_size = args['image_size']
        self.patch_size = args['patch_size']
        self.dims = args['dims']
        self.channels = args['channels']
        self.expansion = args['expansion']
        self.kernel_size = args['kernel_size']
        self.depths = args['depths']
        self.in_channels = args['in_channels']

        assert len(self.dims) == 3, 'dims must be a tuple of 3'
        assert len(self.depths) == 3, 'depths must be a tuple of 3'

        ih, iw = self.image_size
        ph, pw = self.patch_size
        assert ih % ph == 0 and iw % pw == 0

        init_dim, *_, last_dim = self.channels

        self.conv1 = conv_nxn_bn(self.in_channels, init_dim, stride=1)

        self.stem = nn.ModuleList([])
        self.stem.append(MV2Block(self.channels[0], self.channels[1], 1, self.expansion))
        self.stem.append(MV2Block(self.channels[1], self.channels[2], 1, self.expansion))
        self.stem.append(MV2Block(self.channels[2], self.channels[3], 1, self.expansion))
        self.stem.append(MV2Block(self.channels[2], self.channels[3], 1, self.expansion))

        self.trunk = nn.ModuleList([])
        self.trunk.append(nn.ModuleList([
            MV2Block(self.channels[3], self.channels[4], 1, self.expansion),
            MobileViTBlock(self.dims[0], self.depths[0], self.channels[5],
                           self.kernel_size, self.patch_size, int(self.dims[0] * 2))
        ]))

        self.trunk.append(nn.ModuleList([
            MV2Block(self.channels[5], self.channels[6], 1, self.expansion),
            MobileViTBlock(self.dims[1], self.depths[1], self.channels[7],
                           self.kernel_size, self.patch_size, int(self.dims[1] * 4))
        ]))

        self.trunk.append(nn.ModuleList([
            MV2Block(self.channels[7], self.channels[8], 1, self.expansion),
            MobileViTBlock(self.dims[2], self.depths[2], self.channels[9],
                           self.kernel_size, self.patch_size, int(self.dims[2] * 4))
        ]))
        
        # CONDENSES CHANNELS TO LAST CHANNEL ARGUMENT VALUE
        self.condense_channels = nn.Sequential(
            conv_1x1_bn(self.channels[-2], last_dim),
        )
        
        print("Created HFFH_ViT Model")
        
    def num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in model_parameters])

    def forward(self, x):
        x = self.conv1(x)

        for conv in self.stem:
            x = conv(x)

        for conv, attn in self.trunk:
            x = conv(x)
            x = attn(x)

        return self.condense_channels(x)

class HFFH_ViT_S(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.image_size = args['image_size']
        self.patch_size = args['patch_size']
        self.dims = args['dims']
        self.channels = args['channels']
        self.expansion = args['expansion']
        self.kernel_size = args['kernel_size']
        self.depths = args['depths']
        self.in_channels = args['in_channels']

        assert len(self.dims) == 3, 'dims must be a tuple of 3'
        assert len(self.depths) == 3, 'depths must be a tuple of 3'

        ih, iw = self.image_size
        ph, pw = self.patch_size
        assert ih % ph == 0 and iw % pw == 0

        self.conv1 = conv_nxn_bn(self.in_channels, self.channels, stride=1)
        self.trunk = nn.ModuleList([])
        self.trunk.append(nn.ModuleList([
            MV2Block(self.channels, self.channels, 1, self.expansion),
            MobileViTBlock(self.dims[0], self.depths[0], self.channels,
                           self.kernel_size, self.patch_size, int(self.dims[0] * 2))
        ]))
        
        # CONDENSES CHANNELS TO LAST CHANNEL ARGUMENT VALUE
        self.condense_channels = nn.Sequential(
            conv_1x1_bn(self.channels, 1),
        )
        
        print("Created HFFH_ViT_S Model")
        
    def num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in model_parameters])

    def forward(self, x):
        x = self.conv1(x)

        for conv, attn in self.trunk:
            x = conv(x)
            x = attn(x)

        return self.condense_channels(x)