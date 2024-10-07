import torch
import torch.nn as nn
import torch.nn.functional as F


def get_act(act_type = "relu"):
    if act_type == "relu":
        return nn.ReLU(inplace=True)
    
    elif act_type == "lrelu":
        return nn.LeakyReLU(inplace=True)

    elif act_type == "silu":
        return nn.SiLU(inplace=True)
    
    else:
        return nn.ReLU(inplace=True)


class CNNBlock(nn.Module):
    def __init__(self, in_chans, out_chans, activation="lrelu", **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, **kwargs)
        self.norm = nn.BatchNorm2d(num_features=out_chans)
        self.act = get_act(act_type=activation)
    
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Block(nn.Module):
    def __init__(self, in_chans, out_chans, expand_ratio = 0.5, shortcut=False, activation="lrelu"):
        super().__init__()
        hidden_dim = int(out_chans * expand_ratio)
        self.conv_1 = CNNBlock(in_chans, hidden_dim, kernel_size=1, activation=activation)
        self.conv_2 = CNNBlock(hidden_dim, out_chans, kernel_size=3, activation=activation, padding=1)
        self.shortcut = shortcut and in_chans == out_chans

    def forward(self, x):
        shortcut = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        
        if self.shortcut:
            x = x + shortcut

        return x
    

class Residual_Block(nn.Module):
    def __init__(self, in_chans, out_chans, n_block = 1, shortcut = False, activation="lrelu"):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(in_chans=in_chans, out_chans=out_chans, activation=activation, shortcut=shortcut)
            for _ in range(n_block)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        
        return x



class darknet(nn.Module):
    def __init__(self): 
        super().__init__()
        self.conv = CNNBlock(in_chans=3, out_chans=32, kernel_size=3, padding=1)

        self.conv1 = CNNBlock(in_chans=32, out_chans=64, kernel_size=3, stride=2, padding=1)
        self.res1 = Residual_Block(in_chans=64, out_chans=64, n_block=1, shortcut=True)

        self.conv2 = CNNBlock(in_chans=64, out_chans=128, kernel_size=3, stride=2, padding=1)
        self.res2 = Residual_Block(in_chans=128, out_chans=128, n_block=2, shortcut=True)

        self.conv3 = CNNBlock(in_chans=128, out_chans=256, kernel_size=3, stride=2, padding=1)
        self.res3 =  Residual_Block(in_chans=256, out_chans=256, n_block=8, shortcut=True)

        self.conv4 = CNNBlock(in_chans=256, out_chans=512, kernel_size=3, stride=2, padding=1)
        self.res4 = Residual_Block(in_chans=512, out_chans=512, n_block=8, shortcut=True)

        self.conv5 = CNNBlock(in_chans=512, out_chans=1024, kernel_size=3, stride=2, padding=1)
        self.res5 = Residual_Block(in_chans=1024, out_chans=1024, n_block=4, shortcut=False)

    def forward(self, x):
        x = self.conv(x)

        x = self.res1(self.conv1(x))
        
        x = self.res2(self.conv2(x))

        x36 = self.res3(self.conv3(x))

        x61 = self.res4(self.conv4(x36))

        x = self.res5(self.conv5(x61))
        
        return x, x36, x61


class Prediction(nn.Module):
    def __init__(self, in_chans, n_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_chans, in_chans * 2, kernel_size = 3, padding = 1),
            nn.Conv2d(in_chans*2, (n_classes + 5) * 3, kernel_size=1)
        )

        self.num_classes = n_classes

    def forward(self, x):
        output = self.pred(x)

        output = output.view(x.size(0), 3, self.num_classes + 5, x.size(2), x.size(3))
        output = output.permute(0,1,3,4,2)

        return output
    

class Head(nn.Module):
    def __init__(self, in_chans, out_chans, n_block, n_classes, scaled = False):
        super().__init__()
        hidden_dim = out_chans // 2
        self.conv_set = Residual_Block(in_chans, in_chans, n_block=n_block)
        self.conv1 = CNNBlock(in_chans, out_chans, kernel_size=1) 

        self.conv2 = CNNBlock(out_chans, hidden_dim, kernel_size=1)
        self.upsampling = nn.Upsample(scale_factor=2)
       
        self.prediction = Prediction(out_chans, n_classes=n_classes)
        self.scaled = scaled

    def forward(self, x):
        x = self.conv_set(x)
        x = self.conv1(x)
        
        output = self.prediction(x)

        if self.scaled:
            x = self.conv2(x)
            x = self.upsampling(x)
       
        return x, output if self.scaled else output
    

class yolov3(nn.Module):
    def __init__(self, n_classes=80):
        super().__init__()
        self.backbone = darknet()
        self.head1 = Head(in_chans = 1024, out_chans = 512, n_block=3, scaled=True, n_classes=n_classes)
        
        self.head2 = Head(in_chans = 512, out_chans = 256, n_block=3, scaled=True, n_classes=n_classes)
        self.conv21 = CNNBlock(in_chans= 256 * 3, out_chans= 256, kernel_size=1)
        self.conv22 = CNNBlock(in_chans=256, out_chans=512, kernel_size = 3, padding=1)

        self.head3 = Head(in_chans = 256, out_chans = 128, n_block=3, scaled=False, n_classes=n_classes)
        self.conv31 = CNNBlock(in_chans= 128 * 3, out_chans= 128, kernel_size=1)
        self.conv32 = CNNBlock(in_chans=128, out_chans=256, kernel_size = 3, padding=1)

        # self.flatten = nn.Flatten()
        # self.classifier = nn.Linear(in_features=173056, out_features=n_classes)
    
    def forward(self, x):
        x, x36, x61 = self.backbone(x)

        x79, yolo82 = self.head1(x)

        x79 = torch.cat([x79, x61], dim = 1)
        x79 = self.conv21(x79)
        x79 = self.conv22(x79)

        x91, yolo94 = self.head2(x79)
        
        x91 = torch.cat([x91, x36], dim =1)
        x91 = self.conv31(x91)
        x91 = self.conv32(x91)

        yolo106 = self.head3(x91)

        return yolo82, yolo94, yolo106