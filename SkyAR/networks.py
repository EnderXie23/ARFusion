import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
from torchvision import models
from torch.optim import lr_scheduler
import math
import utils
import matplotlib.pyplot as plt
import numpy as np
import os

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bisenetv2 import BiSeNetV2

# Decide which device we want to run on
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PI = math.pi

###############################################################################
# Helper Functions
###############################################################################
class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

def define_G(input_nc, output_nc, ngf, netG, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type='none')
    if netG == 'eff':
        net = DeepLabv3PlusEfficientNet()
    elif netG == 'bisenetv2':
        net = BiSeNetV2FCN()
    elif netG == 'alexnet':
        net = AlexNetFCN(coordconv=True)
    elif netG == 'coord_resnet101':
        net = ResNet50FCN(coordconv=True, size=101)
    elif netG == 'resnet50':
        net = ResNet50FCN()
    elif netG == 'coord_resnet50':
        net = ResNet50FCN(coordconv=True)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)

class AddCoords(nn.Module):
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, y_dim, x_dim = input_tensor.size()

        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2).type_as(input_tensor)
        yy_channel = yy_channel.float() / y_dim
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1)

        ret = torch.cat([input_tensor, yy_channel], dim=1)

        return ret

class CoordConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        in_size = in_channels + 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = AddCoords()(x)
        ret = self.conv(ret)
        return ret


###############################################################################
# Model Definitions
###############################################################################

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      padding=atrous_rates[0], dilation=atrous_rates[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      padding=atrous_rates[1], dilation=atrous_rates[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      padding=atrous_rates[2], dilation=atrous_rates[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        size = x.shape[2:]
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        feat3 = self.branch3(x)
        feat4 = self.branch4(x)
        img_pool = self.image_pool(x)
        img_pool = F.interpolate(img_pool, size=size, mode='bilinear', align_corners=True)
        x = torch.cat([feat1, feat2, feat3, feat4, img_pool], dim=1)
        x = self.project(x)
        return x

class DeepLabv3PlusEfficientNet(nn.Module):
    def __init__(self):
        super(DeepLabv3PlusEfficientNet, self).__init__()
        # Load a pretrained EfficientNet-B0 backbone
        efficientnet = models.efficientnet_b0(pretrained=True)
        self.backbone = efficientnet.features
        # We'll use the output of features[2] as the low-level feature (typically 24 channels)
        self.low_level_idx = 2
        # High-level features come from the final EfficientNet layer (320 channels for EfficientNet-B0)
        in_channels_high = 1280
        
        # ASPP module processes the high-level features
        self.aspp = ASPP(in_channels_high, 256, atrous_rates=[6, 12, 18])
        
        # Low-level feature projection: reduce channels to 48
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU())
        
        # Decoder: fuse ASPP output (upsampled) with low-level features
        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.conv_last = nn.Conv2d(256, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input_size = x.shape[2:]
        low_level_feat = None
        # Pass input through the EfficientNet backbone, capturing low-level features at index 2
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i == self.low_level_idx:
                low_level_feat = x
        high_level_feat = x  # final feature map
        
        # Process high-level features with ASPP
        aspp_out = self.aspp(high_level_feat)
        # Upsample ASPP output to match the low-level feature spatial size
        low_level_size = low_level_feat.shape[2:]
        aspp_up = F.interpolate(aspp_out, size=low_level_size, mode='bilinear', align_corners=True)
        # Project low-level features
        low_level_feat = self.low_level_conv(low_level_feat)
        # Concatenate along channel dimension
        x = torch.cat([aspp_up, low_level_feat], dim=1)
        # Decoder convolutions
        x = self.decoder_conv1(x)
        x = self.decoder_conv2(x)
        # Upsample to the original input resolution
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        x = self.conv_last(x)
        x = self.sigmoid(x)
        return x

class AlexNetFCN(nn.Module):
    def __init__(self, coordconv=False):
        """
        FCN with an AlexNet backbone.
        Args:
            coordconv (bool): If True, use CoordConv in place of standard convolutions.
        """
        super(AlexNetFCN, self).__init__()
        self.coordconv = coordconv
        self.alexnet = models.alexnet(pretrained=True)
        if self.coordconv:
            # Replace the first conv layer with CoordConv2d.
            self.alexnet.features[0] = CoordConv2d(3, 64, kernel_size=11, stride=4, padding=2, bias=False)
        
        self.features = self.alexnet.features
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # Define FPN and prediction layers.
        if self.coordconv:
            self.conv_fpn1 = CoordConv2d(256, 192, kernel_size=3, padding=1)
            self.conv_fpn2 = CoordConv2d(192, 64, kernel_size=3, padding=1)
            self.conv_fpn3 = CoordConv2d(64, 64, kernel_size=3, padding=1)
            self.conv_fpn4 = CoordConv2d(64, 64, kernel_size=3, padding=1)
            self.conv_pred_1 = CoordConv2d(64, 64, kernel_size=3, padding=1)
            self.conv_pred_2 = CoordConv2d(64, 1, kernel_size=3, padding=1)
        else:
            self.conv_fpn1 = nn.Conv2d(256, 192, kernel_size=3, padding=1)
            self.conv_fpn2 = nn.Conv2d(192, 64, kernel_size=3, padding=1)
            self.conv_fpn3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.conv_fpn4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.conv_pred_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.conv_pred_2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Save input spatial size.
        input_size = x.shape[2:]  # (H, W)

        # ----- Feature Extraction -----
        # x4: features from layers 0 to 1.
        x4 = self.features[0:2](x)      # e.g., shape (B, 64, H1, W1)
        # x8: after first max-pooling (layer 2).
        x8 = self.features[2](x4)       # shape (B, 64, H2, W2)
        # x16: from layers 3 to 5.
        x16 = self.features[3:6](x8)      # shape (B, 192, H3, W3)
        # x32: from layers 6 onward.
        x32 = self.features[6:](x16)      # shape (B, 256, H4, W4)

        # ----- FPN Fusion with size matching -----
        # Upsample x32 to match x16 exactly.
        x = F.interpolate(self.relu(self.conv_fpn1(x32)), size=x16.shape[2:], mode='nearest')
        x = x + x16

        # Upsample to match x8.
        x = F.interpolate(self.relu(self.conv_fpn2(x)), size=x8.shape[2:], mode='nearest')
        x = x + x8

        # Upsample to match x4.
        x = F.interpolate(self.relu(self.conv_fpn3(x)), size=x4.shape[2:], mode='nearest')
        x = x + x4

        # An additional FPN layer: upsample toward half the input resolution.
        # (You can adjust this target if you prefer a different scaling.)
        target_size = (input_size[0] // 2, input_size[1] // 2)
        x = F.interpolate(self.relu(self.conv_fpn4(x)), size=target_size, mode='nearest')

        # ----- Prediction Layers: Upsample to the exact input size.
        x = F.interpolate(self.relu(self.conv_pred_1(x)), size=input_size, mode='nearest')
        x = self.sigmoid(self.conv_pred_2(x))
        return x

class BiSeNetV2FCN(nn.Module):
    """
    A wrapper model that uses the provided BiSeNetV2 as a segmentation backbone.
    It preserves the input and output resolution.
    - n_classes: the number of segmentation classes (use 1 for binary segmentation).
    - aux_mode: set to 'eval' (or 'pred') if you want a single segmentation output
      without auxiliary outputs.
    """
    def __init__(self, n_classes=1, aux_mode='eval'):
        super(BiSeNetV2FCN, self).__init__()
        # Instantiate the backbone with the chosen number of classes and aux_mode.
        self.bisenet = BiSeNetV2(n_classes=n_classes, aux_mode=aux_mode)
        # If a single channel output is desired and the head does not include a sigmoid,
        # add one here (for multi-class, you might use softmax or leave as raw logits).
        self.activate = nn.Sigmoid() if n_classes == 1 else nn.Identity()

    def forward(self, x):
        # Forward through the BiSeNetV2 model.
        backbone_out = self.bisenet(x)
        # For modes returning a tuple (e.g. 'eval' returns (logits,)),
        # extract the main logits.
        if isinstance(backbone_out, tuple):
            logits = backbone_out[0]
        else:
            logits = backbone_out
        # Optionally apply activation.
        return self.activate(logits)

class ResNet50FCN(torch.nn.Module):
    def __init__(self, coordconv=False, size=50):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet50FCN, self).__init__()
        if size == 50:
            self.resnet = models.resnet50(pretrained=True)
        elif size == 101:
            self.resnet = models.resnet101(pretrained=True)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2)
        self.coordconv = coordconv

        if coordconv:
            self.conv_in = CoordConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.conv_fpn1 = CoordConv2d(2048, 1024, kernel_size=3, padding=1)
            self.conv_fpn2 = CoordConv2d(1024, 512, kernel_size=3, padding=1)
            self.conv_fpn3 = CoordConv2d(512, 256, kernel_size=3, padding=1)
            self.conv_fpn4 = CoordConv2d(256, 64, kernel_size=3, padding=1)
            self.conv_pred_1 = CoordConv2d(64, 64, kernel_size=3, padding=1)
            self.conv_pred_2 = CoordConv2d(64, 1, kernel_size=3, padding=1)
        else:
            self.conv_fpn1 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
            self.conv_fpn2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
            self.conv_fpn3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
            self.conv_fpn4 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
            self.conv_pred_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.conv_pred_2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        # resnet layers
        if self.coordconv:
            x = self.conv_in(x)
        else:
            x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_4 = self.resnet.layer1(x) # 1/4, in=64, out=64
        x_8 = self.resnet.layer2(x_4) # 1/8, in=64, out=128
        x_16 = self.resnet.layer3(x_8) # 1/16, in=128, out=256
        x_32 = self.resnet.layer4(x_16) # 1/32, in=256, out=512

        # FPN layers
        x = self.upsample(self.relu(self.conv_fpn1(x_32)))
        x = self.upsample(self.relu(self.conv_fpn2(x + x_16)))
        x = self.upsample(self.relu(self.conv_fpn3(x + x_8)))
        x = self.upsample(self.relu(self.conv_fpn4(x + x_4)))

        # output layers
        x = self.upsample(self.relu(self.conv_pred_1(x)))
        x = self.sigmoid(self.conv_pred_2(x))

        return x

