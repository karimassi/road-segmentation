import torch
from torch import nn

class UNet_paper(nn.Module):
  # implementation of u-net neural network architecture as seen in original paper arXiv:1505.04597,
  # but applied with padding in order to achieve matching sizes of input and output images

  # is it better to implement residue block as a function or module?
  class ResBlock(nn.Module):
    def __init__(self, in_c, out_c):
      super().__init__()
      self.in_c = in_c
      self.out_c = out_c
    
    def forward(self, input):
      
      convolution_layer = nn.Sequential(
        nn.BatchNorm2d(self.in_c),
        nn.ReLU(inplace=True),
        # to apply 'same' padding set the value of padding to (kernel_size - 1) / 2
        nn.Conv2d(in_channels=self.in_c, out_channels=self.out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(self.out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=self.out_c, out_channels=self.out_c, kernel_size=3, padding=1),
      )

      residual = input
      conv = convolution_layer(input)
      # this line doesn't work since conv has 32 channels and residual only 3
      output = conv + residual
      
      return output

  def __init__(self, num_channels, num_filters):
    super().__init__()
    self.num_channels = num_channels
    self.num_filters = num_filters

  def conv_layer(self, input, in_c, out_c):
    convolution_layer = nn.Sequential(
        # to apply 'same' padding set the value of padding to (kernel_size - 1) / 2
        nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, padding=1),
    )
    output = convolution_layer(input)
    return output

  def down_sample(self, input, in_c, out_c, max_pool):
    resBlock = self.ResBlock(in_c, out_c)
    conv = resBlock(input)
    pool = max_pool(conv)
    return conv, pool

  def up_sample(self, input, in_c, out_c):
    conv = self.conv_layer(input, in_c, out_c)
    trans_conv = nn.ConvTranspose2d(
        in_channels=out_c,
        out_channels=out_c // 2,
        kernel_size=3,
        stride=(2, 2),
        padding=1,
        output_padding=1
    )
    up_samp = trans_conv(conv)
    return up_samp

  def forward(self, input):
    # define max pooling layer
    max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    # encoder part (downsampling)
    in_c, out_c = self.num_channels, self.num_filters
    conv1, pool1 = self.down_sample(input, in_c, out_c, max_pool)
    in_c = out_c; out_c = 2*out_c
    conv2, pool2 = self.down_sample(pool1, in_c, out_c, max_pool)
    in_c = out_c; out_c = 2*out_c
    conv3, pool3 = self.down_sample(pool2, in_c, out_c, max_pool)
    in_c = out_c; out_c = 2*out_c
    conv4, pool4 = self.down_sample(pool3, in_c, out_c, max_pool)
    
    # middle of the network
    in_c = out_c; out_c = 2*out_c
    conv5 = self.conv_layer(pool4, in_c, out_c)

    # switch values of out_c and in_c to prepare them for upsampling part of the net
    temp = out_c; out_c = in_c; in_c = temp
    # transposed convolution
    trans_conv = nn.ConvTranspose2d(
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
    up_samp1 = trans_conv(conv5)
    concat1 = torch.cat((conv4, up_samp1), 1)
    up_samp2 = self.up_sample(concat1, in_c, out_c)
    in_c = out_c; out_c = out_c // 2
    concat2 = torch.cat((conv3, up_samp2), 1)
    up_samp3 = self.up_sample(concat2, in_c, out_c)
    in_c = out_c; out_c = out_c // 2
    concat3 = torch.cat((conv2, up_samp3), 1)
    up_samp4 = self.up_sample(concat3, in_c, out_c)
    in_c = out_c; out_c = out_c // 2
    concat4 = torch.cat((conv1, up_samp4), 1)
    up_samp5 = self.conv_layer(concat4, in_c, out_c)

    # final layer applies 1x1 convolution to map each num_filters-component
    # feature vector to two classes (0 for background, 1 for road). 
    final_conv = nn.Conv2d(in_channels=(in_c // 2), out_channels=1, kernel_size=1)
    output = final_conv(up_samp5)
    print(output.size())

    sig = nn.Sigmoid()
    output2 = sig(output)

    return output2