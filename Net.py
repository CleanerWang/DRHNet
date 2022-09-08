
import torch.nn
import torch.nn as nn
import numpy as np
import resnet_block

def LeakyRelu(x):

    return torch.nn.LeakyReLU()(x)
    
def OurRelu(x):

    leak = 0.1
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * abs(x) - f2 * x
    
def Friend_relu(x):
    x = torch.nn.ReLU()(x)
    b = 255.0 * torch.ones_like(x)
    return torch.min(x, b)
    
#normalization
def Batch_normalization(X):

    return torch.nn.BatchNorm2d(X).cuda()

#group normalization
def GroupNorm(x,G=32):

    N,C, H, W = x.shape
    groupNorm = torch.nn.GroupNorm(G, C).cuda()
    result = groupNorm(x)

    return result


class DehazeNet(nn.Module):
    def __init__(self):
        super(DehazeNet, self).__init__()
        self.conv1_3 = torch.nn.Conv2d(3, 64, 3, 2, 1)
        self.conv1_5 = torch.nn.Conv2d(3, 32, 5, 2, 2)
        self.conv1_7 = torch.nn.Conv2d(3, 32, 7, 2, 3)
        self.conv2 = torch.nn.Conv2d(128, 256, 3, 2, 1)
        self.conv4 = torch.nn.Conv2d(256, 512, 3, 2, 1)  # 1x69x69x512
        self.conv9 = torch.nn.Conv2d(1024, 512, 1, 1)
        self.conv10 = torch.nn.Conv2d(512, 256, 1, 1, )
        self.conv11 = torch.nn.Conv2d(256, 128, 1, 1)
        self.deconv1 = torch.nn.ConvTranspose2d(512, 512, 1, 1, 0)
        self.deconv2 = torch.nn.ConvTranspose2d(512,256,3,2,1,1)
        self.deconv3 = torch.nn.ConvTranspose2d(256, 128, 3, 2, 1)  # 1x128x275x275
        self.deconv4 = torch.nn.ConvTranspose2d(128, 3, 3, 2, 1, 1)  # 1x3x550x550
    def forward(self, input_X):

        #Multi-scale Convolution
        x_conv1_3 = self.conv1_3(input_X)#1x64x 275 x275
        x_conv1_5 = self.conv1_5(input_X)#1x32x 275 x275
        x_conv1_7 = self.conv1_7(input_X)#1x32x 275 x275

        x_conv1 = torch.cat((x_conv1_3, x_conv1_5, x_conv1_7),1)#1x 128x275 x275
        x_conv1 = GroupNorm(x_conv1)
        x_conv1 = LeakyRelu(x_conv1)

        x_conv2 = self.conv2(x_conv1)#1x256x138x138
        x_conv2 = GroupNorm(x_conv2)
        x_conv2 = LeakyRelu(x_conv2)

        x_conv4 = self.conv4(x_conv2)#1x512x69x69
        x_conv4 = GroupNorm(x_conv4)
        x_conv4 = LeakyRelu(x_conv4)

        x_conv6 = resnet_block.identity_block(x_conv4, 3, 512, [256, 256, 512])
        x_conv7 = resnet_block.identity_block(x_conv6, 3, 512, [256, 256, 512])
        x_conv8 = resnet_block.identity_block(x_conv7, 3, 512, [256, 256, 512])
        x_conv8 = resnet_block.identity_block(x_conv8, 3, 512, [256, 256, 512])
        x_conv8 = resnet_block.identity_block(x_conv8, 3, 512, [256, 256, 512])
        x_conv8 = resnet_block.identity_block(x_conv8, 3, 512, [256, 256, 512])
        x_conv8 = resnet_block.identity_block(x_conv8, 3, 512, [256, 256, 512])


        x_conv9 = self.deconv1(x_conv8)#1x512x69x69
        x_conv9 = GroupNorm(x_conv9)
        x_conv9 = OurRelu(x_conv9)
        x_conv9 = torch.cat((x_conv9,x_conv4),1)


        x_conv9 = self.conv9(x_conv9)
        x_conv9 = GroupNorm(x_conv9)
        x_conv9 = LeakyRelu(x_conv9)#1x512x69x69


        x_conv10 = self.deconv2(x_conv9)#1x256x138x138
        x_conv10 = GroupNorm(x_conv10)
        x_conv10 = OurRelu(x_conv10)
        x_conv10 = torch.cat((x_conv10, x_conv2),1)


        x_conv10 = self.conv10(x_conv10)
        x_conv10 = GroupNorm(x_conv10)
        x_conv10 = LeakyRelu(x_conv10)#1x256x138x138


        x_conv11 = self.deconv3(x_conv10)
        x_conv11 = GroupNorm(x_conv11)
        x_conv11 = OurRelu(x_conv11)
        x_conv11 = torch.cat((x_conv11, x_conv1),1)#1x256x275x275


        x_conv11 = self.conv11(x_conv11)
        x_conv11 = GroupNorm(x_conv11)
        x_conv11 = LeakyRelu(x_conv11)


        x_conv12 = self.deconv4(x_conv11)

        model = torch.add(x_conv12,input_X)
        model = Friend_relu(model)

        return model

