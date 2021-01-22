"""2.5D version of a UNet using a modified ResNet as the encoder."""

import torch
from utils import getCenter, count_parameters
from TorchResnet3dconvs import Bottleneck, BasicBlock


class DecoderBlock(torch.nn.Module):
    # define decoder block for the UNet
    # residuals are all coming from prev_feature_map
    def __init__(self, prev_channel, input_channel, output_channel):
        super(DecoderBlock, self).__init__()
        self.up_sampling = torch.nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        # want to disable bias for conv layers before batch norm but bias may be useful if layer goes through relu first
        self.conv1 = torch.nn.Conv3d(prev_channel + input_channel, output_channel, kernel_size=(1, 3, 3),
                                     stride=(1, 1, 1), padding=(0, 1, 1), bias=True)
        self.bn1 = torch.nn.BatchNorm3d(output_channel)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv3d(output_channel, output_channel, kernel_size=(1, 3, 3), stride=(1, 1, 1),
                                     padding=(0, 1, 1), bias=True)
        self.bn2 = torch.nn.BatchNorm3d(output_channel)


    def forward(self, prev_feature_map, x):
        x = self.up_sampling(x)
        # print(x.size())
        # print(getCenter(prev_feature_map).size())
        x = torch.cat((x, getCenter(prev_feature_map)), dim=1)

        # isn't it a bad idea to put a relu after batchnorm? why does resnet do it?
        # x = self.relu(self.bn1(self.conv1(x)))
        # x = self.relu(self.bn2(self.conv2(x)))

        # trying different arrangement of block
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        return x


class _ResNet3DUNetBasicBlock(torch.nn.Module):
    """Module that creates a 2.5D UNet from a pretrained 2D ResNet.
    resnet3d is a resnet using 3D layers rather than 2D layers.
    input_depth is used to determine the middle convolution layers.
    num_classes is the number of non-background classes"""
    # define the U-Net architecture
    def __init__(self, resnet3d, input_depth=1, num_classes=1):
        # make sure input_depth and num_classes are at least 1
        if input_depth < 1 or num_classes < 1:
            raise AttributeError("Both input_depth and num_classes must be at least 1.")
        # initialize the layers used to define the architecture
        super(_ResNet3DUNetBasicBlock, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        # need to chop off layers 'avgpool' and 'fc' from the resnet encoder
        self.encoder = resnet3d
        self.encoder.requires_grad = False
        #del self.encoder.fc, self.encoder.avgpool  # this is now taken care of in the 3D resnet code

        # need to write mid convs to account for different input data depths
        # these layers are where we go from multi-slice to single slice
        mid_layer_list = []
        if input_depth == 1:
            mid_layer_list.append(torch.nn.Conv3d(512, 512, (1, 3, 3), padding=(0, 1, 1)))
            mid_layer_list.append(torch.nn.BatchNorm3d(512))
            mid_layer_list.append(torch.nn.ReLU(inplace=True))
            mid_layer_list.append(torch.nn.Conv3d(512, 512, (1, 3, 3), padding=(0, 1, 1)))
            mid_layer_list.append(torch.nn.BatchNorm3d(512))
            mid_layer_list.append(torch.nn.ReLU(inplace=True))
        elif input_depth == 2:
            mid_layer_list.append(torch.nn.Conv3d(512, 512, (1, 3, 3), padding=(0, 1, 1)))
            mid_layer_list.append(torch.nn.BatchNorm3d(512))
            mid_layer_list.append(torch.nn.ReLU(inplace=True))
            mid_layer_list.append(torch.nn.Conv3d(512, 512, (2, 3, 3), padding=(0, 1, 1)))
            mid_layer_list.append(torch.nn.BatchNorm3d(512))
            mid_layer_list.append(torch.nn.ReLU(inplace=True))
        elif input_depth == 3:
            mid_layer_list.append(torch.nn.Conv3d(512, 512, (2, 3, 3), padding=(0, 1, 1)))
            mid_layer_list.append(torch.nn.BatchNorm3d(512))
            mid_layer_list.append(torch.nn.ReLU(inplace=True))
            mid_layer_list.append(torch.nn.Conv3d(512, 512, (2, 3, 3), padding=(0, 1, 1)))
            mid_layer_list.append(torch.nn.BatchNorm3d(512))
            mid_layer_list.append(torch.nn.ReLU(inplace=True))
        else: # gradually collapse 3-D information
            # depth 2 convs reduce depth of input by 1, depth 3 reduce it by 2
            odd = bool(input_depth % 2) # true if odd number of input slices, false if even
            remaining_layers = input_depth
            if odd:
                while remaining_layers > 1:
                    # conv3d layers with depth 3 until 1 slice remains
                    mid_layer_list.append(torch.nn.Conv3d(512, 512, 3, padding=(0,1,1)))
                    mid_layer_list.append(torch.nn.BatchNorm3d(512))
                    mid_layer_list.append(torch.nn.ReLU(inplace=True))
                    remaining_layers -= 2
            else: # even number of input slices
                # one conv3d layer with depth 2
                mid_layer_list.append(torch.nn.Conv3d(512, 512, (2, 3, 3), padding=(0,1,1)))
                mid_layer_list.append(torch.nn.BatchNorm3d(512))
                mid_layer_list.append(torch.nn.ReLU(inplace=True))
                remaining_layers -= 1
                while remaining_layers > 1:
                    # conv3d layers with depth 3 until one slice remains
                    mid_layer_list.append(torch.nn.Conv3d(512, 512, 3, padding=(0, 1, 1)))
                    mid_layer_list.append(torch.nn.BatchNorm3d(512))
                    mid_layer_list.append(torch.nn.ReLU(inplace=True))
                    remaining_layers -= 2

        self.mid_conv = torch.nn.Sequential(*mid_layer_list)

        # Resnets are all organized into 4 blocks, downsamples 3 times
        self.up_sample1 = DecoderBlock(256, 512, 512)
        self.up_sample2 = DecoderBlock(128, 512, 256)
        self.up_sample3 = DecoderBlock(64, 256, 128)
        self.up_sample4 = DecoderBlock(64, 128, 64)
        self.up_sample5 = DecoderBlock(3, 64, 64)

        # last convolution and final layer
        if num_classes == 1:
            # sigmoid
            self.last_conv = torch.nn.Conv3d(64, 1, (1, 1, 1), padding=0, stride=1, bias=True)
            # self.out = torch.nn.Sigmoid()
        else:
            # softmax
            # need to add one class for the background
            self.last_conv = torch.nn.Conv3d(64, num_classes+1, (1, 1, 1), padding=0, stride=1, bias=True)
            # self.out = torch.nn.Softmax(dim=1)

        # initializing layers, doing it this (long) way to avoid resetting the pretrained encoder weights
        for m in self.mid_conv.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm3d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

        for m in self.up_sample1.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm3d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

        for m in self.up_sample2.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm3d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

        for m in self.up_sample3.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm3d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

        torch.nn.init.kaiming_normal_(self.last_conv.weight, mode='fan_out', nonlinearity='relu')


    def forward(self, x):
        # define what happens during the forward pass for this U-Net
        identity = x
        x0, x1, x2, x3, x4 = self.encoder(x)
        x = self.mid_conv(x4)
        x = self.up_sample1(x3, x)
        x = self.up_sample2(x2, x)
        x = self.up_sample3(x1, x)
        x = self.up_sample4(x0, x)
        x = self.up_sample5(identity, x)
        x = self.last_conv(x)
        x = self.relu(x)
        # x = self.out(x)  # cutting this out to allow for more flexibility with pre-packaged loss functions
        return x


class _ResNet3DUNetBottleneck(torch.nn.Module):
    def __init__(self, resnet3d, input_depth=1, num_classes=1):
        # make sure input_depth and num_classes are at least 1
        if input_depth < 1 or num_classes < 1:
            raise AttributeError("Both input_depth and num_classes must be at least 1.")
        # initialize the layers used to define the architecture
        super(_ResNet3DUNetBottleneck, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        # need to chop off layers 'avgpool' and 'fc' from the resnet encoder
        self.encoder = resnet3d
        self.encoder.requires_grad = False
        # del self.encoder.fc, self.encoder.avgpool  # this is now taken care of in the 3D resnet code

        # need to write mid convs to account for different input data depths
        # these layers are where we go from multi-slice to single slice
        mid_layer_list = []
        if input_depth == 1:
            mid_layer_list.append(torch.nn.Conv3d(2048, 512, (1, 3, 3), padding=(0, 1, 1)))
            mid_layer_list.append(torch.nn.BatchNorm3d(512))
            mid_layer_list.append(torch.nn.ReLU(inplace=True))
            mid_layer_list.append(torch.nn.Conv3d(512, 512, (1, 3, 3), padding=(0, 1, 1)))
            mid_layer_list.append(torch.nn.BatchNorm3d(512))
            mid_layer_list.append(torch.nn.ReLU(inplace=True))
        elif input_depth == 2:
            mid_layer_list.append(torch.nn.Conv3d(2048, 512, (1, 3, 3), padding=(0, 1, 1)))
            mid_layer_list.append(torch.nn.BatchNorm3d(512))
            mid_layer_list.append(torch.nn.ReLU(inplace=True))
            mid_layer_list.append(torch.nn.Conv3d(512, 512, (2, 3, 3), padding=(0, 1, 1)))
            mid_layer_list.append(torch.nn.BatchNorm3d(512))
            mid_layer_list.append(torch.nn.ReLU(inplace=True))
        elif input_depth == 3:
            mid_layer_list.append(torch.nn.Conv3d(2048, 512, (2, 3, 3), padding=(0, 1, 1)))
            mid_layer_list.append(torch.nn.BatchNorm3d(512))
            mid_layer_list.append(torch.nn.ReLU(inplace=True))
            mid_layer_list.append(torch.nn.Conv3d(512, 512, (2, 3, 3), padding=(0, 1, 1)))
            mid_layer_list.append(torch.nn.BatchNorm3d(512))
            mid_layer_list.append(torch.nn.ReLU(inplace=True))
        else:  # gradually collapse 3-D information
            # depth 2 convs reduce depth of input by 1, depth 3 reduce it by 2
            odd = bool(input_depth % 2)  # true if odd number of input slices, false if even
            remaining_layers = input_depth
            if odd:
                mid_layer_list.append(torch.nn.Conv3d(2048, 512, 3, padding=(0, 1, 1)))
                mid_layer_list.append(torch.nn.BatchNorm3d(512))
                mid_layer_list.append(torch.nn.ReLU(inplace=True))
                remaining_layers -= 2
                while remaining_layers > 1:
                    # conv3d layers with depth 3 until 1 slice remains
                    mid_layer_list.append(torch.nn.Conv3d(512, 512, 3, padding=(0, 1, 1)))
                    mid_layer_list.append(torch.nn.BatchNorm3d(512))
                    mid_layer_list.append(torch.nn.ReLU(inplace=True))
                    remaining_layers -= 2
            else:  # even number of input slices
                # one conv3d layer with depth 2
                mid_layer_list.append(torch.nn.Conv3d(2048, 512, (2, 3, 3), padding=(0, 1, 1)))
                mid_layer_list.append(torch.nn.BatchNorm3d(512))
                mid_layer_list.append(torch.nn.ReLU(inplace=True))
                remaining_layers -= 1
                while remaining_layers > 1:
                    # conv3d layers with depth 3 until one slice remains
                    mid_layer_list.append(torch.nn.Conv3d(512, 512, 3, padding=(0, 1, 1)))
                    mid_layer_list.append(torch.nn.BatchNorm3d(512))
                    mid_layer_list.append(torch.nn.ReLU(inplace=True))
                    remaining_layers -= 2

        self.mid_conv = torch.nn.Sequential(*mid_layer_list)

        # Resnets are all organized into 4 blocks, downsamples 3 times
        self.up_sample1 = DecoderBlock(1024, 512, 512)
        self.up_sample2 = DecoderBlock(512, 512, 256)
        self.up_sample3 = DecoderBlock(256, 256, 128)
        self.up_sample4 = DecoderBlock(64, 128, 64)
        self.up_sample5 = DecoderBlock(3, 64, 64)

        # last convolution and final layer
        if num_classes == 1:
            # sigmoid
            self.last_conv = torch.nn.Conv3d(64, 1, (1, 1, 1), padding=0, stride=1, bias=True)
            # self.out = torch.nn.Sigmoid()
        else:
            # softmax
            # need to add one class for the background
            self.last_conv = torch.nn.Conv3d(64, num_classes + 1, (1, 1, 1), padding=0, stride=1, bias=True)
            # self.out = torch.nn.Softmax(dim=1)

        # initializing layers, doing it this (long) way to avoid resetting the pretrained encoder weights
        for m in self.mid_conv.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm3d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

        for m in self.up_sample1.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm3d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

        for m in self.up_sample2.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm3d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

        for m in self.up_sample3.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm3d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

        torch.nn.init.kaiming_normal_(self.last_conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # define what happens during the forward pass for this U-Net
        identity = x
        x0, x1, x2, x3, x4 = self.encoder(x)
        x = self.mid_conv(x4)
        x = self.up_sample1(x3, x)
        x = self.up_sample2(x2, x)
        x = self.up_sample3(x1, x)
        x = self.up_sample4(x0, x)
        x = self.up_sample5(identity, x)
        x = self.last_conv(x)
        x = self.relu(x)
        # x = self.out(x)  # cutting this out to allow for more flexibility with pre-packaged loss functions
        return x


def ResNet3DUNet(resnet3d, input_depth=1, num_classes=1):
    if isinstance(resnet3d.layer1[0], BasicBlock):
        model = _ResNet3DUNetBasicBlock(resnet3d, input_depth, num_classes)
    elif isinstance(resnet3d.layer1[0], Bottleneck):
        model = _ResNet3DUNetBottleneck(resnet3d, input_depth, num_classes)
    else:
        print("Encoder is not a valid model.")
        model = None
    return model


if __name__ == '__main__':
    from TorchResnet3dconvs import resnet34, resnet18, resnet50
    resnetmodel = resnet34(pretrained=True, is_encoder=True)
    # print(resnetmodel)
    rnmodel = ResNet3DUNet(resnetmodel, input_depth=5, num_classes=2)
    # for m in model.modules():
    #     print(m)
    print(rnmodel)
    # print('resnet params:')
    # count_parameters(resnetmodel)
    # print('unet params:')
    count_parameters(rnmodel)
