import argparse
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn

class DownsampleBlock(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding, bias):
        super(DownsampleBlock, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=dim_in,
                      out_channels=dim_out,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=bias),
            nn.InstanceNorm2d(num_features=dim_out, affine=True),
            nn.GLU(dim=1)
        )
        self.conv_gated = nn.Sequential(
            nn.Conv2d(in_channels=dim_in,
                      out_channels=dim_out,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=bias),
            nn.InstanceNorm2d(num_features=dim_out, affine=True),
            nn.GLU(dim=1)
        )

    def forward(self, x):
        # GLU
        return self.conv_layer(x) * torch.sigmoid(self.conv_gated(x))

class UpSampleBlock(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding, bias):
        super(UpSampleBlock, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=dim_in,
                               out_channels=dim_out,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=bias),
            nn.PixelShuffle(2),
            nn.GLU(dim=1)
        )
        self.conv_gated = nn.Sequential(
            nn.ConvTranspose2d(in_channels=dim_in,
                               out_channels=dim_out,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=bias),
            nn.PixelShuffle(2),
            nn.GLU(dim=1)
        )

    def forward(self, x):
        # GLU
        return self.conv_layer(x) * torch.sigmoid(self.conv_gated(x))

class AdaptiveInstanceNormalization(nn.Module):
    """
        AdaIN block.
    """

    def __init__(self, dim_in, style_num):
        super(AdaptiveInstanceNormalization, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.fc = nn.Linear(style_num, dim_in * 2)

    def forward(self, x, c):
        h = self.fc(c)
        #print (h.shape)
        h = h.view(h.size(0), h.size(1), 1)
        #print (h.shape)
        u = torch.mean(x, dim=2, keepdim=True)
        #print (u.shape)
        var = torch.mean((x - u) * (x - u), dim=2, keepdim=True)
        #print (var.shape)
        std = torch.sqrt(var + 1e-8)
        #print (std.shape)

        gamma, beta = torch.chunk(h, chunks=2, dim=1)#分离出来

        return (1 + gamma) * (x - u) / std + beta

class ConditionalInstanceNormalisation(nn.Module):
    """
        CIN block.
    """

    def __init__(self, dim_in, style_num):
        super(ConditionalInstanceNormalisation, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.dim_in = dim_in
        self.style_num = style_num
        self.gamma = nn.Linear(style_num, dim_in)
        self.beta = nn.Linear(style_num, dim_in)

    def forward(self, x, c):
        u = torch.mean(x, dim=2, keepdim=True)
        var = torch.mean((x - u) * (x - u), dim=2, keepdim=True)
        std = torch.sqrt(var + 1e-8)

        gamma = self.gamma(c.to(self.device))
        gamma = gamma.view(-1, self.dim_in, 1)
        beta = self.beta(c.to(self.device))
        beta = beta.view(-1, self.dim_in, 1)

        h = (x - u) / std
        h = h * gamma + beta

        return h

class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding, style_num):
        super(ResidualBlock, self).__init__()

        self.conv_layer = nn.Conv1d(in_channels=dim_in,
                                    out_channels=dim_out,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding)
        self.adain = AdaptiveInstanceNormalization(dim_in=dim_out, style_num=style_num)
        # self.cin = ConditionalInstanceNormalisation(dim_in=dim_out, style_num=style_num)
        self.glu = nn.GLU(dim=1)

    def forward(self, x, c_):
        x_ = self.conv_layer(x)
        x_ = self.adain(x_, c_)
        x_ = self.glu(x_)

        return x + x_

class Verify_CNN(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding, num_speaker,verityswitch2 = 3):
        super(Verify_CNN, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.verifyingswitch2 = verityswitch2
        # 定义卷积层
        self.conv0_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
        )

        self.conv0_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
        )

        self.conv0_3 = nn.Sequential(
            nn.Conv1d(in_channels=2304, out_channels=256, kernel_size=5, stride=1, padding=2),
        )

        self.conv1 = nn.Conv1d(in_channels=dim_in,
                                out_channels=dim_out,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding)
        self.conv2 = nn.Conv1d(in_channels=dim_in,
                                out_channels=dim_out,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding)
        self.conv3 = nn.Conv1d(in_channels=dim_in,
                                out_channels=dim_out,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding)
        # 定义池化层
        self.pool = nn.MaxPool2d(2, 2)
        self.glu = nn.GLU(dim=1)
        # 定义全连接层
        self.fc1 = nn.Linear(256*32, 500)
        self.fc2 = nn.Linear(500, num_speaker)
        #self.fc3 = nn.Linear(8192, num_speaker)
        #self.flat = nn.Flatten(start_dim=1, end_dim=2)

    def forward(self, x):
        if self.verifyingswitch2 == 1:
            width_size = x.size(3)  # 128
            #print(x.shape)
            x = self.conv0_1(x)
            #print("x:",x.shape)
            x = self.conv0_2(x)
            #print("2x:",x.shape)
            x = x.contiguous().view(x.size(0), 2304, width_size // 4)
            #print(x.shape)
            x = self.conv0_3(x)
            #print("3x:", x.shape)
        elif self.verifyingswitch2 == 2:
            width_size = x.size(3)
            #print(x.shape)
            x = self.conv0_2(x)
            #print("2x:",x.shape)
            x = x.contiguous().view(x.size(0), 2304, width_size // 2)
            #print(x.shape)
            x = self.conv0_3(x)

        x = self.glu(F.relu(self.conv1(x)))
        x = self.glu(F.relu(self.conv2(x)))
        x = self.glu(F.relu(self.conv3(x)))
        # 展平特征
        x = x.view(x.size(0),-1)
        # 通过全连接层
        x = F.relu(self.fc1(x))
        #x = self.fc1(x)
        x = self.fc2(x)
        #x = F.sigmoid(self.fc2(x))
        #print(x.shape)
        return x

    ''' x = self.flat(x)
        x = self.fc3(x)
        return F.softmax(x, dim=-1)
        '''

    def lock_grad(self):
        for param in self.parameters():
            param.requires_grad = False

    def acquire_grad(self):
        for param in self.parameters():
            param.requires_grad = True

class Encoder(nn.Module):
    def __init__(self, num_speakers=4):
        super(Encoder, self).__init__()

        self.num_speakers = num_speakers
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initial layers.
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(5, 15), stride=(1, 1), padding=(2, 7)),
            nn.GLU(dim=1)
        )

        # Down-sampling layers.
        self.down_sample_1 = DownsampleBlock(dim_in=64,
                                             dim_out=256,
                                             kernel_size=(5, 5),
                                             stride=(2, 2),
                                             padding=(2, 2),
                                             bias=False)

        self.down_sample_2 = DownsampleBlock(dim_in=128,
                                             dim_out=512,
                                             kernel_size=(5, 5),
                                             stride=(2, 2),
                                             padding=(2, 2),
                                             bias=False)

        # Reshape data (This operation is done in forward function).

        # Down-conversion layers.
        self.down_conversion = nn.Sequential(
            nn.Conv1d(in_channels=2304,
                      out_channels=256,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.InstanceNorm1d(num_features=256, affine=True)
        )

    def forward(self, x):
        width_size = x.size(3)
        x = self.conv_layer_1(x)
        x = self.down_sample_1(x)
        x = self.down_sample_2(x)
        x = x.contiguous().view(-1, 2304, width_size // 4)
        x = self.down_conversion(x)

        return x, width_size

class GeneratorTop(nn.Module):
    def __init__(self, num_speakers=4):
        super(GeneratorTop, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_speakers = num_speakers

        # Bottleneck layers.
        self.residual_1 = ResidualBlock(dim_in=256,
                                        dim_out=512,
                                        kernel_size=5,
                                        stride=1,
                                        padding=2,
                                        style_num=self.num_speakers * 2)
        self.residual_2 = ResidualBlock(dim_in=256,
                                        dim_out=512,
                                        kernel_size=5,
                                        stride=1,
                                        padding=2,
                                        style_num=self.num_speakers * 2)
        self.residual_3 = ResidualBlock(dim_in=256,
                                        dim_out=512,
                                        kernel_size=5,
                                        stride=1,
                                        padding=2,
                                        style_num=self.num_speakers * 2)
        self.residual_4 = ResidualBlock(dim_in=256,
                                        dim_out=512,
                                        kernel_size=5,
                                        stride=1,
                                        padding=2,
                                        style_num=self.num_speakers * 2)
        self.residual_5 = ResidualBlock(dim_in=256,
                                        dim_out=512,
                                        kernel_size=5,
                                        stride=1,
                                        padding=2,
                                        style_num=self.num_speakers * 2)
        self.residual_6 = ResidualBlock(dim_in=256,
                                        dim_out=512,
                                        kernel_size=5,
                                        stride=1,
                                        padding=2,
                                        style_num=self.num_speakers * 2)
        self.residual_7 = ResidualBlock(dim_in=256,
                                        dim_out=512,
                                        kernel_size=5,
                                        stride=1,
                                        padding=2,
                                        style_num=self.num_speakers * 2)
        self.residual_8 = ResidualBlock(dim_in=256,
                                        dim_out=512,
                                        kernel_size=5,
                                        stride=1,
                                        padding=2,
                                        style_num=self.num_speakers * 2)
        self.residual_9 = ResidualBlock(dim_in=256,
                                        dim_out=512,
                                        kernel_size=5,
                                        stride=1,
                                        padding=2,
                                        style_num=self.num_speakers * 2)

        # 上采样卷积层，包含一个1D卷积层，用于将模型输出转换为与输入相同的形状
        # Up-conversion layers.
        self.up_conversion = nn.Conv1d(in_channels=256,
                                       out_channels=2304,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bias=False)

        # Reshape data (This operation is done in forward function).

        # Up-sampling layers.
        self.up_sample_1 = UpSampleBlock(dim_in=256,
                                         dim_out=1024,
                                         kernel_size=(5, 5),
                                         stride=(1, 1),
                                         padding=2,
                                         bias=False)
        self.up_sample_2 = UpSampleBlock(dim_in=128,
                                         dim_out=512,
                                         kernel_size=(5, 5),
                                         stride=(1, 1),
                                         padding=2,
                                         bias=False)

        self.out = nn.Conv2d(in_channels=64,
                             out_channels=1,
                             kernel_size=(5, 15),
                             stride=(1, 1),
                             padding=(2, 7),
                             bias=False)

        # if self.verifyingswitch:
        #     self.verifyCNN = Verify_CNN(dim_in=256,
        #                                 dim_out=512,
        #                                 kernel_size=5,
        #                                 stride=1,
        #                                 padding=2,
        #                                 num_speaker=4,
        #                                 verityswitch2=3)

    def forward(self, x, c, c_, size):
        c_onehot = torch.cat((c, c_), dim=1).to(self.device)
        width_size = size  # 获取输入语音特征x的宽带
        x = self.residual_1(x, c_onehot)
        x = self.residual_2(x, c_onehot)
        x = self.residual_3(x, c_onehot)
        x = self.residual_4(x, c_onehot)
        x = self.residual_5(x, c_onehot)
        x = self.residual_6(x, c_onehot)
        x = self.residual_7(x, c_onehot)
        x = self.residual_8(x, c_onehot)
        x = self.residual_9(x, c_onehot)
        x = self.up_conversion(x)
        x = x.view(-1, 256, 9, width_size // 4)
        x = self.up_sample_1(x)
        x = self.up_sample_2(x)
        out = self.out(x)
        out_reshaped = out[:, :, : -1, :]

        return out_reshaped


class Generator(nn.Module):
    def __init__(self, num_speakers=4):
        super(Generator, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.encoder = Encoder(num_speakers=num_speakers)
        self.generator_top = GeneratorTop(num_speakers=num_speakers)

    def forward(self, x, c, c_):
        encoded_features, size = self.encoder(x)
        #print(size)
        result = self.generator_top(encoded_features, c, c_, size)
        return result


class Discriminator(nn.Module):
    def __init__(self, num_speakers=4):
        super(Discriminator, self).__init__()

        self.num_speakers = num_speakers
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initial layers.
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.GLU(dim=1)
        )
        self.conv_gated_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.GLU(dim=1)
        )

        # Down-sampling layers.
        self.down_sample_1 = DownsampleBlock(dim_in=64,
                                             dim_out=256,
                                             kernel_size=(3, 3),
                                             stride=(2, 2),
                                             padding=1,
                                             bias=False)
        self.down_sample_2 = DownsampleBlock(dim_in=128,
                                             dim_out=512,
                                             kernel_size=(3, 3),
                                             stride=(2, 2),
                                             padding=1,
                                             bias=False)
        self.down_sample_3 = DownsampleBlock(dim_in=256,
                                             dim_out=1024,
                                             kernel_size=(3, 3),
                                             stride=(2, 2),
                                             padding=1,
                                             bias=False)
        self.down_sample_4 = DownsampleBlock(dim_in=512,
                                             dim_out=1024,
                                             kernel_size=(1, 5),
                                             stride=(1, 1),
                                             padding=(0, 2),
                                             bias=False)

        # Fully connected layer.
        self.fully_connected = nn.Linear(in_features=512, out_features=1)

        # Projection.
        self.projection = nn.Linear(self.num_speakers * 2, 512)

    def forward(self, x, c, c_):
        c_onehot = torch.cat((c, c_), dim=1).to(self.device)
        #print (x.shape)
        x = self.conv_layer_1(x) * torch.sigmoid(self.conv_gated_1(x))
        #print (x.shape)

        x = self.down_sample_1(x)
        #print (x.shape)
        x = self.down_sample_2(x)
        #print (x.shape)
        x = self.down_sample_3(x)
        #print (x.shape)
        x_ = self.down_sample_4(x)
        #print (x.shape)

        h = torch.sum(x_, dim=(2, 3)) # sum pooling
        #print (h.shape)

        x = self.fully_connected(h)

        #print (x.shape)
        p = self.projection(c_onehot)
        #print (p.shape)
        x += torch.sum(p * h, dim=1, keepdim=True)
        #print (x.shape)
        return x

    '''
    # 原论文的方法
    h = torch.sum(x_, dim=(2, 3))
    x = self.fully_connected(x_.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # (b, 1, h, w)
    # print(x.shape)
    p = self.projection(c_onehot)  # (b, 512)
    in_prod = p * h
    x = x.view(x.size(0), -1)
    x = torch.mean(x, dim=-1) + torch.mean(in_prod, dim=-1)
    return x
    '''