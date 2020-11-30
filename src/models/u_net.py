import torch
from torch import nn

class UNet(nn.Module):
  # implementation of u-net neural network architecture as seen in original paper arXiv:1505.04597,
  # but applied with padding in order to achieve matching sizes of input and output images

  def __init__(self, num_channels, num_filters):
    super().__init__()
    self.num_channels = num_channels
    self.num_filters = num_filters

    # define max pooling layer
    self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    # encoder part (downsampling)
    in_c, out_c = self.num_channels, self.num_filters
    self.down_samp1 = self.down_sample(in_c, out_c)
    in_c = out_c; out_c = 2*out_c
    self.down_samp2 = self.down_sample(in_c, out_c)
    in_c = out_c; out_c = 2*out_c
    self.down_samp3 = self.down_sample(in_c, out_c)
    in_c = out_c; out_c = 2*out_c
    self.down_samp4 = self.down_sample(in_c, out_c)
    
    # middle of the network
    in_c = out_c; out_c = 2*out_c
    self.middle_down = self.conv_layer(in_c, out_c)

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

    # decoder part (upsampling)
    self.up_samp1 = self.up_sample(in_c, out_c)
    in_c = out_c; out_c = out_c // 2
    self.up_samp2 = self.up_sample(in_c, out_c)
    in_c = out_c; out_c = out_c // 2
    self.up_samp3 = self.up_sample(in_c, out_c)
    in_c = out_c; out_c = out_c // 2
    self.up_samp4 = self.conv_layer(in_c, out_c)

    # final layer applies 1x1 convolution to map each num_filters-component
    # feature vector to two classes (0 for background, 1 for road). 
    self.final_conv = nn.Conv2d(in_channels=(in_c // 2), out_channels=1, kernel_size=1)
    self.sig = nn.Sigmoid()

  def conv_layer(self, in_c, out_c):
    convolution_layer = nn.Sequential(
        # to apply 'same' padding set the value of padding to (kernel_size - 1) / 2
        nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )
    return convolution_layer

  def down_sample(self, in_c, out_c):
    conv = self.conv_layer(in_c, out_c)
    return conv

  def up_sample(self, in_c, out_c):
    conv = self.conv_layer(in_c, out_c)
    trans_conv = nn.ConvTranspose2d(
        in_channels=out_c,
        out_channels=out_c // 2,
        kernel_size=3,
        stride=(2, 2),
        padding=1,
        output_padding=1
    )
    up_samp = nn.Sequential(
        conv,
        trans_conv
    )
    return up_samp

  def forward(self, input):

    # encoder part (downsampling)
    conv1 = self.down_samp1(input)
    pool1 = self.max_pool(conv1)

    conv2 = self.down_samp2(pool1)
    pool2 = self.max_pool(conv2)

    conv3 = self.down_samp3(pool2)
    pool3 = self.max_pool(conv3)

    conv4 = self.down_samp4(pool3)
    pool4 = self.max_pool(conv4)
    
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

    # final layer applies 1x1 convolution to map each num_filters-component
    # feature vector to two classes (0 for background, 1 for road). 
    output = self.final_conv(up_conv4)
    output2 = self.sig(output)
    #print(output2.size())

    return output2

