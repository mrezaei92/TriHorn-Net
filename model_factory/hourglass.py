import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F
from model_factory.estimationhead import build_EstimationHead


class AdaptiveSpatialSoftmaxLayer(nn.Module):
    def __init__(self,spread=None,train_spread=False,num_channel=14):
        super(AdaptiveSpatialSoftmaxLayer, self).__init__()
        # spread should be a torch tensor of size (1,num_chanel,1)
        # the softmax is applied over spatial dimensions 
        # train determines whether you would like to train the spread parameters as well
        if spread is None:
            self.spread=nn.Parameter(torch.ones(1,num_channel,1))
        else:
            self.spread=nn.Parameter(spread)
        
        self.spread.requires_grad=bool(train_spread)


    def forward(self, x):
        # the input is a tensor of shape (batch,num_channel,height,width)
        SpacialSoftmax = nn.Softmax(dim=2)
        num_batch=x.shape[0]
        num_channel=x.shape[1]
        height=x.shape[2]
        width=x.shape[3]
        inp=x.view(num_batch,num_channel,-1)
        #if self.spread is not None:
        res=torch.mul(inp,self.spread)
        res=SpacialSoftmax(res)
        
        return res.reshape(num_batch,num_channel,height,width)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, BN,num_G,stride=1,downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes) if BN else nn.GroupNorm(num_G, inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes) if BN else nn.GroupNorm(num_G, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes) if BN else nn.GroupNorm(num_G, planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out
    

class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth,BN,num_G):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.BN = BN
        self.num_G = num_G
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth,BN,num_G)

    def _make_residual(self, block, num_blocks, planes,BN,num_G):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes*block.expansion, planes,BN=BN,num_G=num_G))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth,BN,num_G):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes,BN,num_G))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes,BN,num_G))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n-1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n-1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n-1, low1)
        else:
            low2 = self.hg[n-1][3](low1)
        low3 = self.hg[n-1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)
    
    

class HourglassNet(nn.Module):
    '''Hourglass model from Newell et al ECCV 2016'''
    def __init__(self, block, num_stacks=2, num_blocks=4, num_classes=14,BN=True,num_G=16,train_spread=False): #BN ture -> batch normalization, otherwise Group Normalization
        super(HourglassNet, self).__init__()
        #self.sp=nn.Parameter(torch.ones(1,num_classes,1) )
        #self.sp=torch.requires_grad=True
        self.soft0=AdaptiveSpatialSoftmaxLayer(train_spread=train_spread,num_channel=num_classes)#.cuda()#to(device)
        if num_stacks>1:
            self.soft1=AdaptiveSpatialSoftmaxLayer(train_spread=train_spread,num_channel=num_classes)#.cuda()#to(device)
        self.Xs=GetValuesX()
        self.Ys=GetValuesY()
        
        self.BN=BN
        self.num_G=num_G
        self.num_classes = num_classes
        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=1, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes) if BN else nn.GroupNorm(num_G, self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes,1,BN,num_G )
        self.layer2 = self._make_residual(block, self.inplanes, 1,BN,num_G )
        self.layer3 = self._make_residual(block, self.num_feats, 1,BN,num_G)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        ch = self.num_feats*block.expansion
        
        self.hg = Hourglass(block, num_blocks, self.num_feats, 4,BN,num_G)

        self.estimator = build_EstimationHead(3, input_dim =256 ,depth_dim =64, num_classes=num_classes, train_spread=train_spread, BN=BN,num_G=num_G)
        
       


    def _make_residual(self, block, planes, blocks, BN,num_G,stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes,BN,num_G, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,BN,num_G))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes) if self.BN else nn.GroupNorm(self.num_G, inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
                conv,
                bn,
                self.relu,
            )

    def forward(self, x , return_heatmap=False):
        out = []
        outD= []
        self.Xs = self.Xs.to(x.device)
        self.Ys = self.Ys.to(x.device)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        y = self.hg(x)

        
        if return_heatmap:
            UVD,hmp = self.estimator(y,return_heatmap = True)
            return UVD,hmp

        UVD = self.estimator(y)
        return UVD

     


########################################################################################


def GetValuesX(dimension=64,num_channel=14):
    n=dimension
    num_channel=14
    vec=np.linspace(0,dimension-1,dimension).reshape(1,-1)
    Xs=np.linspace(0,dimension-1,dimension).reshape(1,-1)
    for i in range(n-1):
        Xs=np.concatenate([Xs,vec],axis=1)

    #Xs=np.repeat(Xs,num_channel,axis=0)
    Xs=np.float32(np.expand_dims(Xs,axis=0))
    return torch.from_numpy(Xs)

def GetValuesY(dimension=64,num_channel=14):
    res=np.zeros((1,dimension*dimension))
    for i in range(dimension):
        res[0,(i*dimension):((i+1)*dimension)]=i
    res=np.float32( np.expand_dims(res,axis=0) )
    return torch.from_numpy(res)
