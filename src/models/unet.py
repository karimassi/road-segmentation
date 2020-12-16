import torch
from torch import nn

class UNet(nn.Module):
    # implementation of u-net neural network with residual blocks as seen in paper doi:10.3390/app9224825

    def __init__(self, num_channels, num_filters):
        super().__init__()
        self.num_channels = num_channels
        self.num_filters = num_filters

        # define encoder part (downsampling)
        in_c, out_c = self.num_channels, self.num_filters
        self.down_samp1 = DownSample(in_c, out_c)
        in_c = out_c; out_c = 2*out_c
        self.down_samp2 = DownSample(in_c, out_c)
        in_c = out_c; out_c = 2*out_c
        self.down_samp3 = DownSample(in_c, out_c)

        # define middle of the network
        in_c = out_c; out_c = 2*out_c
        self.middle_down = ResBlock(in_c, out_c)

        # switch values of out_c and in_c to prepare them for upsampling part of the net
        temp = out_c; out_c = in_c; in_c = temp
        # transposed convolution for upsampling
        self.middle_up = nn.ConvTranspose2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=3,
            stride=(2, 2),
            padding=1,
            # when stride > 1, convolution maps multiple input shapes to the same output shape
            # we use output_padding to obtain correct output shape of transposed convolution
            output_padding=1
        )

        # define decoder part (upsampling)
        self.up_samp1 = UpSample(in_c, out_c)
        in_c = out_c; out_c = out_c // 2
        self.up_samp2 = UpSample(in_c, out_c)
        in_c = out_c; out_c = out_c // 2
        self.up_samp3 = ConvLayer(in_c, out_c)

        # final layer applies 1x1 convolution to map each num_filters-component
        # feature vector to two classes ([0, 1] for background, [1, 0] for road).
        self.final_conv = nn.Conv2d(in_channels=(in_c // 2), out_channels=2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):

        # encoder part (downsampling)
        conv1, pool1 = self.down_samp1(input)
        conv2, pool2 = self.down_samp2(pool1)
        conv3, pool3 = self.down_samp3(pool2)

        # middle of the network
        mid_down = self.middle_down(pool3)
        mid_up = self.middle_up(mid_down)

        # decoder part (upsampling)
        concat1 = torch.cat((conv3, mid_up), 1)
        up_conv1 = self.up_samp1(concat1)

        concat2 = torch.cat((conv2, up_conv1), 1)
        up_conv2 = self.up_samp2(concat2)

        concat3 = torch.cat((conv1, up_conv2), 1)
        up_conv3 = self.up_samp3(concat3)

        # end of network
        final = self.final_conv(up_conv3)
        output = self.sigmoid(final)

        return output

class DownSample(nn.Module):
    def __init__(self, num_channels, num_filters):
        super().__init__()
        self.num_channels = num_channels
        self.num_filters = num_filters

        self.conv = ResBlock(num_channels, num_filters)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, input):
        conv_out = self.conv(input)
        max_out = self.max_pool(conv_out)
        return conv_out, max_out

class UpSample(nn.Module):
    def __init__(self, num_channels, num_filters):
        super().__init__()
        self.num_channels = num_channels
        self.num_filters = num_filters

        conv = ConvLayer(num_channels, num_filters)
        trans_conv = nn.ConvTranspose2d(
            in_channels=num_filters,
            out_channels=num_filters // 2,
            kernel_size=3,
            stride=(2, 2),
            padding=1,
            output_padding=1
        )
        self.up_samp = nn.Sequential(
            conv,
            trans_conv
        )

    def forward(self, input):
        return self.up_samp(input)

class ConvLayer(nn.Module):
    def __init__(self, num_channels, num_filters):
        super().__init__()
        self.num_channels = num_channels
        self.num_filters = num_filters

        self.layer = nn.Sequential(
            # to apply 'same' padding set the value of padding to (kernel_size - 1) / 2
            nn.Conv2d(in_channels=num_channels, out_channels=num_filters, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, padding=1),
        )

    def forward(self, input):
        return self.layer(input)

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c

        self.convolution_layer = nn.Sequential(
          nn.BatchNorm2d(self.in_c),
          nn.ReLU(inplace=True),
          # to apply 'same' padding set the value of padding to (kernel_size - 1) / 2
          nn.Conv2d(in_channels=self.in_c, out_channels=self.out_c, kernel_size=3, padding=1),
          nn.BatchNorm2d(self.out_c),
          nn.ReLU(inplace=True),
          nn.Conv2d(in_channels=self.out_c, out_channels=self.out_c, kernel_size=3, padding=1),
        )

        # increasing number of channels of original input to match dimensionality with conv(x)
        self.match_dim = nn.Conv2d(in_channels=self.in_c, out_channels=self.out_c, kernel_size=3, padding=1)

    def forward(self, input):

        residual = input
        conv = self.convolution_layer(input)
        residual = self.match_dim(residual)

        output = conv + residual

        return output
