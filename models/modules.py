# flake8: noqa
import os
import sys
sys.path.append(os.getcwd())
import math
import torch
from torch.nn import init

import torch.nn.functional as F
from einops import rearrange
import torch.nn as nn
from basicsr.archs.arch_util import LayerNorm2d

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None
    
class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
    
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)
    if hasattr(m, 'bias') and m.bias is not None:
        init.constant_(m.bias.data, 0.0)

# @ARCH_REGISTRY.register()
class MSBE(nn.Module):
    def __init__(self, in_chan, patch_size, scale_level=4, module_num=8,  module_chan=64, middle_feat=15, num_class=3, num_experts=5, use_bias=True):
        super(MSBE, self).__init__()

        self.patch_size = patch_size
        self.padder_size = 2 ** (scale_level - 1)
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_chan, module_chan, kernel_size=3, stride=1,padding=1, bias=use_bias),
            nn.ReLU()
        )
        backbone = []
        for _ in range(module_num):
            backbone.append(nn.Conv2d(module_chan, module_chan, kernel_size=3, stride=1, padding=1, bias=use_bias))
            backbone.append(nn.ReLU())

        self.backbone = nn.Sequential(*backbone)

        clasifiers = []
        weight_heads = []
        for i in range(scale_level):
            clasifiers.append(nn.Sequential(
                nn.AvgPool2d(kernel_size=int(patch_size * math.pow(2, i))),
                nn.Conv2d(module_chan, middle_feat, kernel_size=1,bias=use_bias),
                nn.Conv2d(middle_feat, num_class, kernel_size=1,bias=use_bias)
                
            ))

            weight_heads.append(nn.Sequential(
                nn.Conv2d(num_class, middle_feat, kernel_size=1,bias=use_bias),
                nn.Conv2d(middle_feat, num_experts, kernel_size=1,bias=use_bias)
            ))

        self.clasifiers = nn.Sequential(*clasifiers)
        self.weight_heads = nn.Sequential(*weight_heads)
        self.scale_levle = scale_level
        self.apply(weights_init_kaiming)
    
    

    # def check_image_size(self, x):
    #     n, c, h, w = x.size()
    #     pad_h, pad_w = 0, 0
    #     max_path_size = self.padder_size * self.patch_size
    #     if h % max_path_size != 0:
    #         pad_h = int(max_path_size - h % max_path_size) 
    #     if w % max_path_size != 0:
    #         pad_w = int(max_path_size - w % max_path_size)
        
    #     x = F.pad(x, (0, pad_w, 0, pad_h))

    #     return x

    def forward(self, x):
        # x = self.check_image_size(x)
        x = self.conv_in(x)
        x0 = self.backbone(x)
        clas = []
        vs = []
        for clasifier, weight_head in zip(self.clasifiers, self.weight_heads):
            x = clasifier(x0)
            v = weight_head(x)
            clas.append(x)
            vs.append(v)
        return clas, vs


class PMoEConv(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size, stride=1, padding=1, dilation=1, patch_size=64, bias=True, K = 5):
        super(PMoEConv, self).__init__()

        self.K = K
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.patch_size = patch_size
        self.weight = nn.Parameter(torch.randn(K, out_chan, in_chan // self.groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_chan), requires_grad=True)
        else:
            self.bias = None

        self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])
            if self.bias is not None:
                nn.init.constant_(self.bias[i], 0)

    def forward(self, x, v):
        """
        x 输入特征图
        v 加权
        """
        _, _, h, w = x.shape
        y = rearrange(x, 'b c (h p1) (w p2) -> (b h w) c p1 p2', p1=self.patch_size, p2=self.patch_size)
        v = rearrange(v, 'b k h w -> (b h w) k')

        batch_size, in_planes, height, width = y.size()
        y = y.contiguous().view(1, -1, height, width) # 1, b * in_chan, height, width
        weight = self.weight.view(self.K, -1) # K, n

        aggregate_weight = torch.mm(v, weight).view(-1, in_planes, self.kernel_size , self.kernel_size) # b * out_chan, in_chan, k, k

        if self.bias is not None:
            aggregate_bias = torch.mm(v, self.bias).view(-1)
            output = F.conv2d(y, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups * batch_size)
        else:
            output = F.conv2d(y, weight=aggregate_weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_chan, output.size(-2), output.size(-1))
        output = rearrange(output, '(b h w) c p1 p2 -> b c (h p1) (w p2)',h=h//self.patch_size, w = w//self.patch_size)
        return output
    

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class PMoECB(nn.Module):
    def __init__(self, c, patch_size, K=5, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = PMoEConv(in_chan=c, out_chan=dw_channel, patch_size=patch_size, kernel_size=1, padding=0, stride=1, bias=True, K=K)
        # self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = PMoEConv(in_chan=dw_channel, out_chan=dw_channel, patch_size=patch_size, kernel_size=3,padding=1,stride=1,bias=True, K=K)
        # self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
        #                        bias=True)
        self.conv3 = PMoEConv(in_chan=dw_channel//2, out_chan=c,patch_size=patch_size,kernel_size=1,padding=0,stride=1,bias=True, K=K)
        # self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.sca = PMoEConv(in_chan=dw_channel//2, out_chan=dw_channel//2, patch_size=patch_size, kernel_size=1,padding=0,stride=1,bias=True, K=K)


        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = PMoEConv(in_chan=c,out_chan=ffn_channel, patch_size=patch_size, kernel_size=1,padding=0,stride=1,bias=True, K=K)
        # self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.conv5 = PMoEConv(in_chan=ffn_channel//2, out_chan=c, patch_size=patch_size, kernel_size=1,padding=0,stride=1,bias=True, K=K)
        # self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inps):
        inp, v = inps
        x = inp

        x = self.norm1(x)
        
        x = self.conv1(x, v)
        x = self.conv2(x, v)
        keep_x = self.sg(x)
        
        x = self.avg(keep_x)
        x = x * self.sca(keep_x, v)
        x = self.conv3(x, v)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y), v)
        x = self.sg(x)
        x = self.conv5(x, v)

        x = self.dropout2(x)

        return y + x * self.gamma, v

# @ARCH_REGISTRY.register()   
class BAEM(nn.Module):

    def __init__(self, img_channel=3, out_chanel = 320, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], patch_size = 32, K =5):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.patch_size = patch_size

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[PMoECB(chan, patch_size=patch_size,K=K) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[PMoECB(chan, patch_size=patch_size,K=K) for _ in range(middle_blk_num)]
            )
        self.conv_out = zero_conv_module(
             nn.Conv2d(chan, out_chanel, kernel_size=3, stride=1, padding=1)
        )
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[PMoECB(chan, patch_size=patch_size,K=K) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp, vs):
        B, C, H, W = inp.shape
        # inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down, v in zip(self.encoders, self.downs, vs):
            # print(x.shape, v.shape)
            x, _ = encoder([x, v])
            encs.append(x)
            x = down(x)

        x, _ = self.middle_blks([x, vs[-1]])
        x0 = self.conv_out(x)

        vs = vs[:-1]
        for decoder, up, enc_skip, v in zip(self.decoders, self.ups, encs[::-1], vs[::-1]):
            x = up(x)

            x = x + enc_skip
            x, _ = decoder([x, v])
        x = self.ending(x)
        x = x + inp

        return x0, x[:, :, :H, :W]

    # def check_image_size(self, x):
    #     n, c, h, w = x.size()
    #     pad_h, pad_w = 0, 0
    #     max_path_size = self.padder_size * self.patch_size
    #     if h % max_path_size != 0:
    #         pad_h = int(max_path_size - h % max_path_size) 
    #     if w % max_path_size != 0:
    #         pad_w = int(max_path_size - w % max_path_size)
        
    #     x = F.pad(x, (0, pad_w, 0, pad_h))
    #     return x

def zero_conv_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module   

if __name__ == '__main__':
    patch_size = 64
    num_class = 4 # clear, artifact, defocus, color distortion
    K = 4 # 四种专家
    patch_level = 4

    device = torch.device('cuda')
    ce_model = ClarityEstimator(3, patch_size, patch_level, 8, 32, 15, num_class, K, True).to(device=device)
    model = PCAMoENet(3, width=32, enc_blk_nums=[1,1,1], middle_blk_num=1, dec_blk_nums=[1, 1, 1], patch_size = patch_size, K=K).to(device=device)


    print(next(ce_model.clasifiers.parameters()).device)

    # width: 64
    # enc_blk_nums: [1, 1, 1, 28]
    # middle_blk_num: 1
    # dec_blk_nums: [1, 1, 1, 1]
    # dep_blk_chans: [6, 64, 128, 256, 128, 64, 1]

    # with torch.no_grad():

    x = torch.randn(1,3, 768, 768).to(device=device)
    cs, vs = ce_model(x)


    # print(cs,vs)
    torch.cuda.empty_cache()
    # print(cs, vs)
    # print(x.shape, v.shape)

    for c, v in zip(cs, vs):
        print(c.shape, v.shape)

    x0, out = model(x, vs = vs)
    print(x0.shape)

