import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

##############################################
# Helper functions for initialization, etc.
##############################################

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        return nn.BatchNorm2d
    elif norm_type == 'instance':
        return nn.InstanceNorm2d
    else:
        # no normalization – identity function
        return lambda x: x

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    # (A dummy initialization: you can add your own weight initialization here)
    if gpu_ids and torch.cuda.is_available():
        net.to(gpu_ids[0])
    return net

##############################################
# CoordConv helper modules
##############################################

class AddCoords(nn.Module):
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape (batch, channel, y_dim, x_dim)
        """
        batch_size, _, y_dim, x_dim = input_tensor.size()

        # Create a normalized y-coordinate channel
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2).type_as(input_tensor)
        yy_channel = yy_channel.float() / y_dim
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1)

        # Optionally, you could also add a radial distance channel if with_r==True

        ret = torch.cat([input_tensor, yy_channel], dim=1)
        return ret

class CoordConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        # Increase in_channels by 1 due to the added coordinate channel.
        in_size = in_channels + 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        x = AddCoords()(x)
        x = self.conv(x)
        return x

##############################################
# EfficientNet Backbone for DeepLabv3+
##############################################

class EfficientNetBackbone(nn.Module):
    def __init__(self, backbone_name="efficientnet_b0", pretrained=True, coordconv=False):
        """
        Loads an EfficientNet backbone and splits it into two parts:
          - low_level features: taken from an early block (here, after features[2])
          - high_level features: the remainder (which typically outputs the head features)
        """
        super(EfficientNetBackbone, self).__init__()
        self.coordconv = coordconv
        self.efficientnet = models.efficientnet_b0(pretrained=pretrained)
        self.features = self.efficientnet.features
        
        # For EfficientNet-B0 (torchvision), the sequential 'features' typically contains 9 modules.
        # We choose the output after features[2] as the low-level feature.
        # (For EfficientNet-B0, features[2] usually outputs 24 channels.)
        self.low_level_channels = 24
        # The full backbone output is expected to have 1280 channels.
        self.high_level_channels = 1280
        
        # If coordconv is enabled, replace the very first conv layer in features[0]
        if coordconv:
            # features[0] is typically a Sequential; we assume its first module is the conv layer.
            conv0 = self.features[0][0]
            in_channels = conv0.in_channels
            out_channels = conv0.out_channels
            kernel_size = conv0.kernel_size
            stride = conv0.stride
            padding = conv0.padding
            bias = (conv0.bias is not None)
            self.features[0][0] = CoordConv2d(in_channels, out_channels, kernel_size=kernel_size,
                                              stride=stride, padding=padding, bias=bias)
            
    def forward(self, x):
        low_level_feat = None
        # Run through each block; save the output after block index 2 as low-level features.
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 2:
                low_level_feat = x
        high_level_feat = x
        return low_level_feat, high_level_feat

##############################################
# ASPP module for DeepLabv3+
##############################################

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        """
        Atrous Spatial Pyramid Pooling.
        Args:
            in_channels: number of input channels (from the high-level features)
            out_channels: number of output channels for each branch (commonly 256)
            atrous_rates: list of dilation rates for the 3x3 convolutions
        """
        super(ASPP, self).__init__()
        modules = []
        # 1x1 convolution branch.
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        # 3x3 convolution branches with different atrous rates.
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        # Image-level pooling branch.
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
    def forward(self, x):
        res = []
        # Run the first branches.
        for conv in self.convs[:-1]:
            res.append(conv(x))
        # Global pooling branch.
        global_feat = self.convs[-1](x)
        global_feat = F.interpolate(global_feat, size=x.size()[2:], mode='bilinear', align_corners=False)
        res.append(global_feat)
        # Concatenate and project.
        x = torch.cat(res, dim=1)
        return self.project(x)

##############################################
# Decoder module for DeepLabv3+
##############################################

class Decoder(nn.Module):
    def __init__(self, low_level_in_channels, low_level_out_channels, num_classes):
        """
        The decoder module fuses the low-level features (from early layers) with the ASPP output.
        """
        super(Decoder, self).__init__()
        self.conv_low_level = nn.Sequential(
            nn.Conv2d(low_level_in_channels, low_level_out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_level_out_channels),
            nn.ReLU(inplace=True)
        )
        # After concatenation, the channel count is (low_level_out_channels + ASPP_out_channels).
        self.last_conv = nn.Sequential(
            nn.Conv2d(low_level_out_channels + 256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
    def forward(self, low_level_feat, high_level_feat):
        low_level_feat = self.conv_low_level(low_level_feat)
        # Upsample high-level features to match the spatial size of low-level features.
        high_level_feat = F.interpolate(high_level_feat, size=low_level_feat.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat([low_level_feat, high_level_feat], dim=1)
        return self.last_conv(x)

##############################################
# DeepLabv3+ with EfficientNet backbone
##############################################

class DeepLabv3PlusEfficientNet(nn.Module):
    def __init__(self, num_classes=1, backbone_name="efficientnet_b0", pretrained_backbone=True, coordconv=False):
        """
        Implements DeepLabv3+ using an EfficientNet backbone.
        Args:
            num_classes: number of segmentation classes (use 1 for binary segmentation)
            backbone_name: which EfficientNet variant to use (e.g. "efficientnet_b0")
            pretrained_backbone: whether to load pretrained weights for the backbone
            coordconv: if True, use coordinate convolution for the very first layer.
        """
        super(DeepLabv3PlusEfficientNet, self).__init__()
        self.backbone = EfficientNetBackbone(backbone_name, pretrained=pretrained_backbone, coordconv=coordconv)
        # ASPP module applied on the high-level features.
        self.aspp = ASPP(in_channels=self.backbone.high_level_channels, out_channels=256, atrous_rates=[6, 12, 18])
        # Decoder module: here we reduce low-level features to 48 channels.
        self.decoder = Decoder(low_level_in_channels=self.backbone.low_level_channels, low_level_out_channels=48, num_classes=num_classes)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        input_size = x.size()[2:]
        low_level_feat, high_level_feat = self.backbone(x)
        x = self.aspp(high_level_feat)
        x = self.decoder(low_level_feat, x)
        # Upsample to the original image size.
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        x = self.sigmoid(x)
        return x

##############################################
# define_G function (generator definition)
##############################################

def define_G(input_nc, output_nc, ngf, netG, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
    This function instantiates the generator network.
    You can switch between DeepLabv3+ and its coordconv variant by using:
       - netG == 'deeplabv3plus' 
       - netG == 'coord_deeplabv3plus'
    """
    net = None
    norm_layer = get_norm_layer(norm_type='none')
    if netG == 'deeplabv3plus':
        net = DeepLabv3PlusEfficientNet(num_classes=output_nc, backbone_name="efficientnet_b0", 
                                         pretrained_backbone=True, coordconv=False)
    elif netG == 'coord_deeplabv3plus':
        net = DeepLabv3PlusEfficientNet(num_classes=output_nc, backbone_name="efficientnet_b0", 
                                         pretrained_backbone=True, coordconv=True)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)
