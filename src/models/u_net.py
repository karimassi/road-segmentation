import torch
from torch import nn

class UNet(nn.Module):
  # implementation of u-net neural network architecture as seen in original paper arXiv:1505.04597,
  # but applied with padding in order to achieve matching sizes of input and output images

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
    in_c = out_c; out_c = 2*out_c
    self.down_samp4 = DownSample(in_c, out_c)
    
    # define middle of the network
    in_c = out_c; out_c = 2*out_c
    self.middle_down = ConvLayer(in_c, out_c)

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
    self.up_samp3 = UpSample(in_c, out_c)
    in_c = out_c; out_c = out_c // 2
    self.up_samp4 = ConvLayer(in_c, out_c)

    # final layer applies 1x1 convolution to map each num_filters-component
    # feature vector to two classes ([0, 1] for background, [1, 0] for road).
    self.final_conv = nn.Conv2d(in_channels=(in_c // 2), out_channels=2, kernel_size=1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, input):

    # encoder part (downsampling)
    conv1, pool1 = self.down_samp1(input)
    conv2, pool2 = self.down_samp2(pool1)
    conv3, pool3 = self.down_samp3(pool2)
    conv4, pool4 = self.down_samp4(pool3)
    
    # middle of the network
    mid_down = self.middle_down(pool4)
    mid_up = self.middle_up(mid_down)

    # decoder part (upsampling)
    concat1 = torch.cat((conv4, mid_up), 1)
    up_conv1 = self.up_samp1(concat1)
    
    concat2 = torch.cat((conv3, up_conv1), 1)
    up_conv2 = self.up_samp2(concat2)
    
    concat3 = torch.cat((conv2, up_conv2), 1)
    up_conv3 = self.up_samp3(concat3)
    
    concat4 = torch.cat((conv1, up_conv3), 1)
    up_conv4 = self.up_samp4(concat4)

    # end of network
    final = self.final_conv(up_conv4)
    output = self.sigmoid(final)

    return output

class DownSample(nn.Module):
  def __init__(self, num_channels, num_filters):
    super().__init__()
    self.num_channels = num_channels
    self.num_filters = num_filters

    self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv = ConvLayer(num_channels, num_filters)

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
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

  def forward(self, input):
    return self.layer(input)

