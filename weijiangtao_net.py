import torch
from torch import nn
import torchvision
import torch.nn.functional as F
class Reshape(nn.Module):
    def forward(self,x):
        return x.view(-1,1,28,28)

class weijiangtao_net(nn.Module):
    def __init__(self):
        super(weijiangtao_net,self).__init__()
        self.Conv1 = nn.Sequential(
            #第一个卷积层，输入通道为1，输出通道为6，卷积核大小为3，步长为1，填充为1保证输入输出尺寸相同
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1),#padding = 1保证输入输出尺寸相同
            #激活函数，两个网络层之间加入，引入非线性
            nn.ReLU(),
            #池化层，大小为2步长为2
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Flatten()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(12*6*6,160),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(160,80),
            nn.ReLU()
        )
        #最后一层全连接层神经元数目10,与上一个全连接层同理
        self.fc3 = nn.Linear(80,10)
    def forward(self,input):
        output = self.Conv1(input)

        output = self.Conv2(output)

        output = self.fc1(output)

        output = self.fc2(output)

        output = self.fc3(output)

        return output

# AlexNet
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 48, 11, 4,2),  # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
              # kernel_size, stride
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(48, 128, 5, 1,2 ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(128, 192, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(192,192,3,1,1),
            nn.ReLU(),
            nn.Conv2d(192, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            nn.Flatten()

        )
        # 这里全连接层的输出个数比LeNet中的大数倍。使用丢弃层来缓解过拟合
        self.fc = nn.Sequential(

            nn.Linear(4608, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048,1000),

            # 输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 2),
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class MyAlexNet(nn.Module):
    def __init__(self):
        super(MyAlexNet, self).__init__()
        self.c1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, stride=4, padding=2)
        self.ReLU = nn.ReLU()
        self.c2 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.s2 = nn.MaxPool2d(2)
        self.c3 = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.s3 = nn.MaxPool2d(2)
        self.c4 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.s5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.f6 = nn.Linear(4608, 2048)
        self.f7 = nn.Linear(2048, 2048)
        self.f8 = nn.Linear(2048, 1000)
        self.f9 = nn.Linear(1000, 2)

    def forward(self, x):
        x = self.ReLU(self.c1(x))
        x = self.ReLU(self.c2(x))
        x = self.s2(x)
        x = self.ReLU(self.c3(x))
        x = self.s3(x)
        x = self.ReLU(self.c4(x))
        x = self.ReLU(self.c5(x))
        x = self.s5(x)
        x = self.flatten(x)
        x = self.f6(x)
        x = F.dropout(x, p=0.5)
        x = self.f7(x)
        x = F.dropout(x, p=0.5)
        x = self.f8(x)
        x = F.dropout(x, p=0.5)

        x = self.f9(x)
        return x

#U-net
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # up-conv 2*2
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, high, low):
        x1 = self.up(high)
        offset = x1.size()[2] - low.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        # 计算应该填充多少（这里可以是负数）
        x2 = F.pad(low, padding)  # 这里相当于对低级特征做一个crop操作
        x1 = torch.cat((x1, x2), dim=1)  # 拼起来
        x1 = self.conv_relu(x1)  # 卷积走起
        return x1

#卷积核个数，卷积次数减少，padding加起来
class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 1024, 3,padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3,padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.decorder4 = Decoder(1024, 512)
        self.decorder3 = Decoder(512, 256)
        self.decorder2 = Decoder(256, 128)
        self.decorder1 = Decoder(128, 64)
        self.last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        # Encorder
        layer1 = self.layer1(input)
        layer2 = self.layer2(self.maxpool(layer1))
        layer3 = self.layer3(self.maxpool(layer2))
        layer4 = self.layer4(self.maxpool(layer3))
        layer5 = self.layer5(self.maxpool(layer4))

        # Decorder
        layer6 = self.decorder4(layer5, layer4)
        layer7 = self.decorder3(layer6, layer3)
        layer8 = self.decorder2(layer7, layer2)
        layer9 = self.decorder1(layer8, layer1)
        out = self.last(layer9)  # n_class预测种类数

        return out
#MyUNet
class MyUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # # self.layer4 = nn.Sequential(
        # #     nn.Conv2d(128, 256, 3,padding=1),
        # #     nn.BatchNorm2d(256),
        # #     nn.ReLU(inplace=True),
        # #     nn.Conv2d(256, 256, 3,padding=1),
        # #     nn.BatchNorm2d(256),
        # #     nn.ReLU(inplace=True)
        # )
        # self.layer5 = nn.Sequential(
        #     nn.Conv2d(256, 512, 3,padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, 3,padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True)
        # )

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.decorder4 = Decoder(512, 256)
        self.decorder3 = Decoder(256, 128)
        self.decorder2 = Decoder(128, 64)
        self.decorder1 = Decoder(64, 32)
        self.last = nn.Conv2d(32, n_class, 1)

    def forward(self, input):
        # Encorder
        layer1 = self.layer1(input)
        layer2 = self.layer2(self.maxpool(layer1))
        layer3 = self.layer3(self.maxpool(layer2))
        # layer4 = self.layer4(self.maxpool(layer3))
        # layer5 = self.layer5(self.maxpool(layer4))

        # Decorder
        # layer6 = self.decorder4(layer5, layer4)
        # layer7 = self.decorder3(layer4, layer3)
        layer8 = self.decorder2(layer3, layer2)
        layer9 = self.decorder1(layer8, layer1)
        out = self.last(layer9)  # n_class预测种类数

        return out
#MyUNet2
class UNet1(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, 3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, 3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.decorder4 = Decoder(512, 256)
        self.decorder3 = Decoder(256, 128)
        self.decorder2 = Decoder(128, 64)
        self.decorder1 = Decoder(64, 32)
        self.last = nn.Conv2d(32, n_class, 3,padding=1)

    def forward(self, input):
        # Encorder
        layer1 = self.layer1(input)
        layer2 = self.layer2(self.maxpool(layer1))
        layer3 = self.layer3(self.maxpool(layer2))
        layer4 = self.layer4(self.maxpool(layer3))
        layer5 = self.layer5(self.maxpool(layer4))

        # Decorder
        layer6 = self.decorder4(layer5, layer4)
        layer7 = self.decorder3(layer6, layer3)
        layer8 = self.decorder2(layer7, layer2)
        layer9 = self.decorder1(layer8, layer1)
        out = self.last(layer9)  # n_class预测种类数

        return out

