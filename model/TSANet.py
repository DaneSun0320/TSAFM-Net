import math

import torch
from torch import nn
from torch.nn import functional as F

Conv2d = nn.Conv2d

IN_CHANNLS = 1


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=3, dilation=3, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=5, dilation=5, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet18(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet18-5c106cde.pth
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)

class InitialBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 main_branch_out_channel,
                 bias=False,
                 relu=True):
        super().__init__()

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # Main branch - As stated above the number of output channels for this
        # branch is the total minus 3, since the remaining channels come from
        # the extension branch
        self.main_branch = nn.Conv2d(
            in_channels,
            main_branch_out_channel,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=bias)

        # Extension branch
        self.ext_branch = nn.MaxPool2d(3, stride=2, padding=1)

        # Initialize batch normalization to be used after concatenation
        self.batch_norm = nn.BatchNorm2d(out_channels)

        self.out_stem = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()

    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)

        # Concatenate branches
        out = torch.cat((main, ext), 1)

        # Apply batch normalization
        out = self.batch_norm(out)
        out = self.out_activation(out)

        return self.out_stem(out)


class RegularBottleneck(nn.Module):
    """Regular bottlenecks are the main building block of ENet.
    Main branch:
    1. Shortcut connection.

    Extension branch:
    1. 1x1 convolution which decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. regular, dilated or asymmetric convolution;
    3. 1x1 convolution which increases the number of channels back to
    ``channels``, also called an expansion;
    4. dropout as a regularizer.

    Keyword arguments:
    - channels (int): the number of input and output channels.
    - internal_ratio (int, optional): a scale factor applied to
    ``channels`` used to compute the number of
    channels after the projection. eg. given ``channels`` equal to 128 and
    internal_ratio equal to 2 the number of channels after the projection
    is 64. Default: 4.
    - kernel_size (int, optional): the kernel size of the filters used in
    the convolution layer described above in item 2 of the extension
    branch. Default: 3.
    - padding (int, optional): zero-padding added to both sides of the
    input. Default: 0.
    - dilation (int, optional): spacing between kernel elements for the
    convolution described in item 2 of the extension branch. Default: 1.
    asymmetric (bool, optional): flags if the convolution described in
    item 2 of the extension branch is asymmetric or not. Default: False.
    - dropout_prob (float, optional): probability of an element to be
    zeroed. Default: 0 (no dropout).
    - bias (bool, optional): Adds a learnable bias to the output if
    ``True``. Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.

    """

    def __init__(self,
                 channels,
                 internal_ratio=4,
                 kernel_size=3,
                 padding=0,
                 dilation=1,
                 asymmetric=False,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()

        # Check in the internal_scale parameter is within the expected range
        # [1, channels]
        if internal_ratio <= 1 or internal_ratio > channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}."
                               .format(channels, internal_ratio))

        internal_channels = channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # Main branch - shortcut connection

        # Extension branch - 1x1 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution, and,
        # finally, a regularizer (spatial dropout). Number of channels is constant.

        # 1x1 projection convolution
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                channels,
                internal_channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(internal_channels), activation())

        # If the convolution is asymmetric we split the main convolution in
        # two. Eg. for a 5x5 asymmetric convolution we have two convolution:
        # the first is 5x1 and the second is 1x5.
        if asymmetric:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(kernel_size, 1),
                    stride=1,
                    padding=(padding, 0),
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation(),
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(1, kernel_size),
                    stride=1,
                    padding=(0, padding),
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())
        else:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(channels), activation())

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after adding the branches
        self.out_activation = activation()

    def forward(self, x):
        # Main branch shortcut
        main = x

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = main + ext

        return self.out_activation(out)


class DownsamplingBottleneck(nn.Module):
    """Downsampling bottlenecks further downsample the feature map size.

    Main branch:
    1. max pooling with stride 2; indices are saved to be used for
    unpooling later.

    Extension branch:
    1. 2x2 convolution with stride 2 that decreases the number of channels
    by ``internal_ratio``, also called a projection;
    2. regular convolution (by default, 3x3);
    3. 1x1 convolution which increases the number of channels to
    ``out_channels``, also called an expansion;
    4. dropout as a regularizer.

    Keyword arguments:
    - in_channels (int): the number of input channels.
    - out_channels (int): the number of output channels.
    - internal_ratio (int, optional): a scale factor applied to ``channels``
    used to compute the number of channels after the projection. eg. given
    ``channels`` equal to 128 and internal_ratio equal to 2 the number of
    channels after the projection is 64. Default: 4.
    - return_indices (bool, optional):  if ``True``, will return the max
    indices along with the outputs. Useful when unpooling later.
    - dropout_prob (float, optional): probability of an element to be
    zeroed. Default: 0 (no dropout).
    - bias (bool, optional): Adds a learnable bias to the output if
    ``True``. Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_ratio=4,
                 return_indices=False,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()

        # Store parameters that are needed later
        self.return_indices = return_indices

        # Check in the internal_scale parameter is within the expected range
        # [1, channels]
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}. "
                               .format(in_channels, internal_ratio))

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_max1 = nn.MaxPool2d(
            2,
            stride=2,
            return_indices=return_indices)

        # Extension branch - 2x2 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution. Number
        # of channels is doubled.

        # 2x2 projection convolution with stride 2
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                internal_channels,
                kernel_size=2,
                stride=2,
                bias=bias), nn.BatchNorm2d(internal_channels), activation())

        # Convolution
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                internal_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias), nn.BatchNorm2d(internal_channels), activation())

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(out_channels), activation())

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()

    def forward(self, x):
        # Main branch shortcut
        if self.return_indices:
            main, max_indices = self.main_max1(x)
        else:
            main = self.main_max1(x)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Main branch channel padding
        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n, ch_ext - ch_main, h, w)

        # Before concatenating, check if main is on the CPU or GPU and
        # convert padding accordingly
        if main.is_cuda:
            padding = padding.to(main.device)

        # Concatenate
        main = torch.cat((main, padding), 1)

        # Add main and extension branches
        out = main + ext

        return self.out_activation(out), max_indices


class UpsamplingBottleneck(nn.Module):
    """The upsampling bottlenecks upsample the feature map resolution using max
    pooling indices stored from the corresponding downsampling bottleneck.

    Main branch:
    1. 1x1 convolution with stride 1 that decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. max unpool layer using the max pool indices from the corresponding
    downsampling max pool layer.

    Extension branch:
    1. 1x1 convolution with stride 1 that decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. transposed convolution (by default, 3x3);
    3. 1x1 convolution which increases the number of channels to
    ``out_channels``, also called an expansion;
    4. dropout as a regularizer.

    Keyword arguments:
    - in_channels (int): the number of input channels.
    - out_channels (int): the number of output channels.
    - internal_ratio (int, optional): a scale factor applied to ``in_channels``
     used to compute the number of channels after the projection. eg. given
     ``in_channels`` equal to 128 and ``internal_ratio`` equal to 2 the number
     of channels after the projection is 64. Default: 4.
    - dropout_prob (float, optional): probability of an element to be zeroed.
    Default: 0 (no dropout).
    - bias (bool, optional): Adds a learnable bias to the output if ``True``.
    Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_ratio=4,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()

        # Check in the internal_scale parameter is within the expected range
        # [1, channels]
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}. "
                               .format(in_channels, internal_ratio))

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels))

        # Remember that the stride is the same as the kernel_size, just like
        # the max pooling layers
        self.main_unpool1 = nn.MaxUnpool2d(kernel_size=2)

        # Extension branch - 1x1 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution. Number
        # of channels is doubled.

        # 1x1 projection convolution with stride 1
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, internal_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(internal_channels), activation())

        # Transposed convolution
        self.ext_tconv1 = nn.ConvTranspose2d(
            internal_channels,
            internal_channels,
            kernel_size=2,
            stride=2,
            bias=bias)
        self.ext_tconv1_bnorm = nn.BatchNorm2d(internal_channels)
        self.ext_tconv1_activation = activation()

        # 1x1 expansion convolution
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                internal_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels))

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()

    def forward(self, x, max_indices, output_size):
        # Main branch shortcut
        main = self.main_conv1(x)
        main = self.main_unpool1(
            main.float(), max_indices, output_size=output_size)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_tconv1(ext, output_size=output_size)
        ext = self.ext_tconv1_bnorm(ext)
        ext = self.ext_tconv1_activation(ext)
        ext = self.ext_conv2(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = main + ext

        return self.out_activation(out)


class ENet(nn.Module):
    """Generate the ENet model.

    Keyword arguments:
    - num_classes (int): the number of classes to segment.
    - encoder_relu (bool, optional): When ``True`` ReLU is used as the
    activation function in the encoder blocks/layers; otherwise, PReLU
    is used. Default: False.
    - decoder_relu (bool, optional): When ``True`` ReLU is used as the
    activation function in the decoder blocks/layers; otherwise, PReLU
    is used. Default: True.

    """

    def __init__(self, in_channels=1, encoder_relu=True, decoder_relu=True):
        super().__init__()

        self.initial_block = InitialBlock(in_channels, 32, 32 - in_channels, relu=encoder_relu)

        # Stage 1 - Encoder
        self.downsample1_0 = DownsamplingBottleneck(
            32,
            64,
            return_indices=True,
            dropout_prob=0.2,
            relu=encoder_relu)
        self.regular1_1 = RegularBottleneck(
            64, padding=1, dropout_prob=0.2, relu=encoder_relu)
        self.regular1_2 = RegularBottleneck(
            64, padding=1, dropout_prob=0.2, relu=encoder_relu)
        self.regular1_3 = RegularBottleneck(
            64, padding=1, dropout_prob=0.2, relu=encoder_relu)
        self.regular1_4 = RegularBottleneck(
            64, padding=1, dropout_prob=0.2, relu=encoder_relu)

        # Stage 2 - Encoder
        self.downsample2_0 = DownsamplingBottleneck(
            64,
            128,
            return_indices=True,
            dropout_prob=0.2,
            relu=encoder_relu)
        self.regular2_1 = RegularBottleneck(
            128, padding=1, dropout_prob=0.2, relu=encoder_relu)
        self.dilated2_2 = RegularBottleneck(
            128, dilation=2, padding=2, dropout_prob=0.2, relu=encoder_relu)
        self.asymmetric2_3 = RegularBottleneck(
            128,
            kernel_size=5,
            padding=2,
            asymmetric=True,
            dropout_prob=0.2,
            relu=encoder_relu)
        self.dilated2_4 = RegularBottleneck(
            128, dilation=4, padding=4, dropout_prob=0.2, relu=encoder_relu)
        self.regular2_5 = RegularBottleneck(
            128, padding=1, dropout_prob=0.2, relu=encoder_relu)
        self.dilated2_6 = RegularBottleneck(
            128, dilation=8, padding=8, dropout_prob=0.2, relu=encoder_relu)
        self.asymmetric2_7 = RegularBottleneck(
            128,
            kernel_size=5,
            asymmetric=True,
            padding=2,
            dropout_prob=0.2,
            relu=encoder_relu)
        self.dilated2_8 = RegularBottleneck(
            128, dilation=16, padding=16, dropout_prob=0.2, relu=encoder_relu)

        # Stage 3 - Encoder
        self.regular3_0 = RegularBottleneck(
            128, padding=1, dropout_prob=0.2, relu=encoder_relu)
        self.dilated3_1 = RegularBottleneck(
            128, dilation=2, padding=2, dropout_prob=0.2, relu=encoder_relu)
        self.asymmetric3_2 = RegularBottleneck(
            128,
            kernel_size=5,
            padding=2,
            asymmetric=True,
            dropout_prob=0.2,
            relu=encoder_relu)
        self.dilated3_3 = RegularBottleneck(
            128, dilation=4, padding=4, dropout_prob=0.2, relu=encoder_relu)
        self.regular3_4 = RegularBottleneck(
            128, padding=1, dropout_prob=0.2, relu=encoder_relu)
        self.dilated3_5 = RegularBottleneck(
            128, dilation=8, padding=8, dropout_prob=0.2, relu=encoder_relu)
        self.asymmetric3_6 = RegularBottleneck(
            128,
            kernel_size=5,
            asymmetric=True,
            padding=2,
            dropout_prob=0.2,
            relu=encoder_relu)
        self.dilated3_7 = RegularBottleneck(
            128, dilation=16, padding=16, dropout_prob=0.2, relu=encoder_relu)

        # Stage 4 - Decoder
        self.upsample4_0 = UpsamplingBottleneck(
            128, 64, dropout_prob=0.2, relu=decoder_relu)
        self.regular4_1 = RegularBottleneck(
            64, padding=1, dropout_prob=0.2, relu=decoder_relu)
        self.regular4_2 = RegularBottleneck(
            64, padding=1, dropout_prob=0.2, relu=decoder_relu)

        # Stage 5 - Decoder
        self.upsample5_0 = UpsamplingBottleneck(
            64, 32, dropout_prob=0.2, relu=decoder_relu)
        self.regular5_1 = RegularBottleneck(
            32, padding=1, dropout_prob=0.2, relu=decoder_relu)

    def forward(self, x):
        outs = []
        # Initial block
        x = self.initial_block(x)

        x_level_5 = x.clone()

        # Stage 1 - Encoder
        stage1_input_size = x.size()
        x, max_indices1_0 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)
        x_level_4 = x.clone()

        # Stage 2 - Encoder
        stage2_input_size = x.size()
        x, max_indices2_0 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)
        x_level_8 = x.clone()

        # Stage 3 - Encoder
        x = self.regular3_0(x)
        x = self.dilated3_1(x)
        x = self.asymmetric3_2(x)
        x = self.dilated3_3(x)
        x = self.regular3_4(x)
        x = self.dilated3_5(x)
        x = self.asymmetric3_6(x)
        x = self.dilated3_7(x)
        x = x + x_level_8

        outs = (x,)
        # Stage 4 - Decoder
        x = self.upsample4_0(x, max_indices2_0, output_size=stage2_input_size)
        x = self.regular4_1(x)
        x = self.regular4_2(x)
        x = x + x_level_4

        outs = outs + (x,)
        # Stage 5 - Decoder
        x = self.upsample5_0(x, max_indices1_0, output_size=stage1_input_size)
        x = self.regular5_1(x) + x_level_5

        outs = outs + (x,)
        return outs

def Conv1x1BnRelu(in_channels, out_channels):
    return nn.Sequential(
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )


def downSampling_2x(in_channels, out_channels):
    return nn.Sequential(
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )


def downSampling_4x(in_channels, out_channels):
    return nn.Sequential(
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.ReLU6(inplace=True),
    )


def downSampling_8x(in_channels, out_channels):
    return nn.Sequential(
        downSampling_4x(in_channels, out_channels),
        downSampling_2x(out_channels, out_channels)
    )


def upSampling_2x(in_channels, out_channels):
    return nn.Sequential(
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True).float(),
        nn.Upsample(scale_factor=2, mode='nearest').float()
    )


def upSampling_4x(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1,
                           output_padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True).float(),
        nn.Upsample(scale_factor=2, mode='nearest').float()
    )


def upSampling_8x(in_channels, out_channels):
    return nn.Sequential(
        upSampling_2x(in_channels, out_channels),
        upSampling_4x(out_channels, out_channels)
    )

class AttentionModule(nn.Module):
    def __init__(self, in_channels, num_heads=2):
        super(AttentionModule, self).__init__()
        self.num_heads = num_heads
        self.q_layer = nn.Linear(in_channels, in_channels)
        self.k_layer = nn.Linear(in_channels, in_channels)
        self.v_layer = nn.Linear(in_channels, in_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(),
        )

    def attention(self, q, k, v, d_k):
        '''
        input:
            q = [d_model,d_model]
            k = [d_model,d_model]
            v = [d_model,d_model]
        output:
            output = [d_model,d_model]
        '''
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        scores = torch.softmax(scores, dim=-1)
        output = torch.matmul(scores, v)  # [d_model,de_model]

        return output


    def _add_relative_position_encoding(self, features):
        # 给特征图增加三角函数编码
        b, hw, c = features.size()
        pos = torch.arange(hw, device=features.device).float() / hw
        dim_t = torch.arange(c, device=features.device).float() / c
        dim_t = 1 - 2 * dim_t
        dim_t = dim_t.view(1, 1, c)
        pos = pos.view(1, hw, 1)
        sin = torch.sin(pos * dim_t)
        cos = torch.cos(pos * dim_t)
        features = features + sin
        features = features + cos
        return features

    def forward(self, x1, x2):
        # f1,f2下采样到1/4
        x1_tmp = F.max_pool2d(x1, kernel_size=4)
        x2_tmp = F.max_pool2d(x2, kernel_size=4)

        # 计算注意力图
        b, c, h, w = x1_tmp.size()
        x1_tmp = x1_tmp.view(b, c, -1)  # [batch_size, channels, height*width]
        x2_tmp = x2_tmp.view(b, c, -1)  # [batch_size, channels, height*width]

        x1_tmp = x1_tmp.permute(0, 2, 1)  # [batch_size, height*width, channels]
        x2_tmp = x2_tmp.permute(0, 2, 1)  # [batch_size, height*width, channels]

        x1_tmp = self._add_relative_position_encoding(x1_tmp)
        x2_tmp = self._add_relative_position_encoding(x2_tmp)

        x1_q_embed = self.q_layer(x1_tmp)
        x1_k_embed = self.k_layer(x2_tmp)
        x1_v_embed = self.v_layer(x1_tmp)
        x1_q_embed = x1_q_embed.reshape(b, h*w, self.num_heads, c // self.num_heads).transpose(1, 2)
        x1_k_embed = x1_k_embed.reshape(b, h*w, self.num_heads, c // self.num_heads).transpose(1, 2)
        x1_v_embed = x1_v_embed.reshape(b, h*w, self.num_heads, c // self.num_heads).transpose(1, 2)

        x2_q_embed = self.q_layer(x2_tmp)
        x2_k_embed = self.k_layer(x1_tmp)
        x2_v_embed = self.v_layer(x2_tmp)
        x2_q_embed = x2_q_embed.reshape(b, h*w, self.num_heads, c // self.num_heads).transpose(1, 2)
        x2_k_embed = x2_k_embed.reshape(b, h*w, self.num_heads, c // self.num_heads).transpose(1, 2)
        x2_v_embed = x2_v_embed.reshape(b, h*w, self.num_heads, c // self.num_heads).transpose(1, 2)



        attention_x1 = self.attention(x1_q_embed, x2_k_embed, x1_v_embed, c // self.num_heads).transpose(1,2).contiguous().view(b,-1, c)
        attention_x2 = self.attention(x2_q_embed, x1_k_embed, x2_v_embed, c // self.num_heads).transpose(1,2).contiguous().view(b,-1, c)

        # 对x1应用注意力图
        x1_att_f = self.conv(attention_x1.transpose(1, 2).view(b, c, h, w))
        x2_att_f = self.conv(attention_x2.transpose(1, 2).view(b, c, h, w))

        x1_att_f = F.interpolate(x1_att_f.float(), scale_factor=2, mode='bilinear', align_corners=True) + F.max_pool2d(x1, kernel_size=2)
        x2_att_f = F.interpolate(x2_att_f.float(), scale_factor=2, mode='bilinear', align_corners=True) + F.max_pool2d(x2, kernel_size=2)

        return x1_att_f, x2_att_f

class ASFF(nn.Module):
    def __init__(self, level, channel1, channel2, channel3, out_channel):
        super(ASFF, self).__init__()
        self.level = level
        funsed_channel = 8

        if self.level == 1:
            # level = 1:
            self.level2_1 = downSampling_2x(channel2, channel1)
            self.level3_1 = downSampling_4x(channel3, channel1)

            self.weight1 = Conv1x1BnRelu(channel1, funsed_channel)
            self.weight2 = Conv1x1BnRelu(channel1, funsed_channel)
            self.weight3 = Conv1x1BnRelu(channel1, funsed_channel)

            self.expand_conv = Conv1x1BnRelu(channel1, out_channel)

        if self.level == 2:
            #  level = 2:
            self.level1_2 = upSampling_2x(channel1, channel2).float()
            self.level3_2 = downSampling_2x(channel3, channel2)

            self.weight1 = Conv1x1BnRelu(channel2, funsed_channel)
            self.weight2 = Conv1x1BnRelu(channel2, funsed_channel)
            self.weight3 = Conv1x1BnRelu(channel2, funsed_channel)

            self.expand_conv = Conv1x1BnRelu(channel2, out_channel)

        if self.level == 3:
            #  level = 3:
            self.level1_3 = upSampling_4x(channel1, channel3).float()
            self.level2_3 = upSampling_2x(channel2, channel3).float()

            self.weight1 = Conv1x1BnRelu(channel3, funsed_channel)
            self.weight2 = Conv1x1BnRelu(channel3, funsed_channel)
            self.weight3 = Conv1x1BnRelu(channel3, funsed_channel)

            self.expand_conv = Conv1x1BnRelu(channel3, out_channel)

        self.weight_level = Conv2d(funsed_channel * 3, 3, kernel_size=1, stride=1, padding=0)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, level1, level2, level3):
        if self.level == 1:
            level_1 = level1.clone()
            level_2 = self.level2_1(level2.float())
            level_3 = self.level3_1(level3.float())

        if self.level == 2:
            level_1 = self.level1_2(level1.float())
            level_2 = level2.clone()
            level_3 = self.level3_2(level3.float())

        if self.level == 3:
            level_1 = self.level1_3(level1.float())
            level_2 = self.level2_3(level2.float())
            level_3 = level3.clone()

        weight1 = self.weight1(level_1)
        weight2 = self.weight2(level_2)
        weight3 = self.weight3(level_3)

        level_weight = torch.cat((weight1, weight2, weight3), dim=1)
        weight_level = self.weight_level(level_weight)
        weight_level = self.softmax(weight_level)

        fused_level = level_1 * weight_level[:, 0, :, :].unsqueeze(1) + level_2 * weight_level[:, 1, :, :].unsqueeze(
            1) + level_3 * weight_level[:, 2, :, :].unsqueeze(1)
        out = self.expand_conv(fused_level)
        return out

class SDP(nn.Module):
    """
    input: (batch_size, 24, h/2, w/2),
            (batch_size, 32, h/4, w/4),
            (batch_size, 56, h/8, w/8)
    output: (batch_size, 24, h/2, w/2),
            (batch_size, 32, h/4, w/4),
            (batch_size, 56, h/8, w/8)
    """

    def __init__(self, in_channels, num_heads=2, out_channel_base=8):
        super(SDP, self).__init__()
        self.attention_module_list = nn.ModuleList()
        self.asff_list = nn.ModuleList()
        for i, c in enumerate(in_channels):
            self.attention_module_list.append(AttentionModule(c, num_heads))
            self.asff_list.append(ASFF(level=i+1, channel1=in_channels[0] * 2, channel2=in_channels[1] * 2, channel3=in_channels[2] * 2, out_channel=out_channel_base // 2**i))

    def forward(self, x):
        tmp = ()
        for i, f in enumerate(x):
            f0, f1 = f
            f0_att, f1_att = self.attention_module_list[i](f0, f1)
            f_i = torch.cat([f0_att, f1_att], dim=1)
            tmp = (f_i,) + tmp
        out = ()
        tmp = tmp[::-1]
        for i in range(3):
            out_i = self.asff_list[i](*tmp)
            out += (out_i,)

        return out

class PreAlign(nn.Module):
    """
    input: (batch_size, 1, h, w),(batch_size, 1, h, w)
    output: (batch_size, 1, h, w)
    """

    def __init__(self):
        super(PreAlign, self).__init__()
        # 定位网络-卷积层
        encoder = resnet18(include_top=False)
        old_kernel_weight = encoder.conv1.weight
        tmp_k = torch.sum(old_kernel_weight, dim=1, keepdim=True)
        tmp_k = tmp_k.repeat(1, 2, 1, 1)
        encoder.conv1.weight = torch.nn.Parameter(tmp_k)
        encoder.conv1.in_channels = 2
        self.localization_convs = nn.Sequential(
            encoder,
            nn.AdaptiveAvgPool2d(output_size=(5, 5)),
            nn.Flatten(1)
        )
        # 定位网络-线性层
        self.localization_linear = nn.Sequential(
            nn.Linear(in_features=512 * 25, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=2 * 3)
        )
        # 初始化定位网络仿射矩阵的权重/偏置，即是初始化θ值。使得图片的空间转换从原始图像开始。
        self.localization_linear[2].weight.data.zero_()
        self.localization_linear[2].bias.data.copy_(torch.tensor([1, 0, 0,
                                                                  0, 1, 0], dtype=torch.float))

    # 空间变换器网络，转发图片张量
    def stn(self, fix, moving):
        x = torch.cat([fix, moving], dim=1)
        # 使用CNN对图像结构定位，生成变换参数矩阵θ（2*3矩阵)
        x = self.localization_convs(x)
        x = self.localization_linear(x)
        theta = x.view(x.size()[0], 2, 3)  # [1, 2, 3]
        '''
        2D空间变换初始θ参数应置为tensor([[[1., 0., 0.],
                                        [0., 1., 0.]]])
        '''
        # 网格生成器，根据θ建立原图片的坐标仿射矩阵
        grid = nn.functional.affine_grid(theta, moving.size(), align_corners=True)
        # 采样器，根据网格对原图片进行转换，转发给CNN分类网络
        x = nn.functional.grid_sample(moving, grid, align_corners=True)
        return theta, x

    def forward(self, fix, moving):
        return self.stn(fix, moving)

class RegressionHead(nn.Module):
    def __init__(self, in_channels, out_nums) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 7, stride=2, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(10, 10)),
            nn.Flatten(1),
            nn.Linear(in_channels * 100, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_nums)
        )

    def forward(self, x):
        return self.fc(x)


class HomographyTransform(nn.Module):

    def __init__(self, base_channel=8, out_size=(50, 50)):
        super(HomographyTransform, self).__init__()
        self.channel_adj = nn.ModuleList([])
        for i in range(3):
            self.channel_adj.append(
                RegressionHead(in_channels=base_channel * 2 ** (3 - i), out_nums=out_size[0] * out_size[1] * 2)
                )

    def forward(self, x):
        b, c, h, w = x[0].shape
        out = 0
        for i in range(3):
            out = out + self.channel_adj[i](x[i])
        return out


class TSA_Net(nn.Module):
    """
    Template-based spatial deformation perception alignment Network
    """

    def __init__(self, in_channel, out_channel_base, out_size):
        super(TSA_Net, self).__init__()
        self.pre_align = PreAlign()
        self.backbone = ENet(in_channels=in_channel)
        self.sdp = SDP(in_channels=[128, 64, 32], num_heads=8, out_channel_base=out_channel_base * 2 ** 3)
        self.homography_transform = HomographyTransform(base_channel=out_channel_base, out_size=out_size)

    def forward(self, t, d):
        theta, d_new = self.pre_align(t, d)
        f_t, f_d = self.backbone(t), self.backbone(d_new)
        f_d = self.sdp(tuple(zip(f_t, f_d)))
        out = self.homography_transform(f_d)
        return theta, d_new, out

    def get_model_name(self):
        return "TSANet"




