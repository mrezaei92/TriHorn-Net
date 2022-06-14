import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F



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
    



class EstimationHead(nn.Module):
    def __init__(self,num_blocks, block, input_dim =256 ,depth_dim =64, num_classes=14, attention_num_blocks=3,train_spread=False, BN=True,num_G=16):
        super(EstimationHead, self).__init__()

        self.num_feats = int(input_dim/2)
        self.input_dim = input_dim
        self.depth_dim = depth_dim
        self.num_classes = num_classes
        
        ch = self.num_feats*2
        
        self.BN = BN
        self.relu = nn.ReLU(inplace=True)

        self.soft = AdaptiveSpatialSoftmaxLayer(train_spread=train_spread,num_channel=num_classes)

        self.register_buffer("Xs", GetValuesX() )
        self.register_buffer("Ys", GetValuesY() )

        self.UVbranch = nn.Sequential( self._make_residual(block, input_dim, self.num_feats, num_blocks,BN,num_G), self._make_fc(ch, ch), nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))

        self.DepthBranch = nn.Sequential( self._make_residual(block, input_dim, self.num_feats, num_blocks,BN,num_G), self._make_fc(ch, ch), nn.Conv2d(ch, self.depth_dim, kernel_size=1, bias=True))

        self.AttentionEnhBranch = nn.Sequential( self._make_residual(block, input_dim, self.num_feats, attention_num_blocks,BN,num_G), self._make_fc(ch, ch), nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))


        self.depth_estimator = nn.Linear(self.depth_dim,1)

        self.Betas =torch.nn.Parameter(torch.ones(1,num_classes,1,1)*0.5)





    def _make_residual(self, block, inplanes, planes, blocks, BN,num_G,stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential( nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=True), )

        layers = []
        layers.append(block(inplanes, planes,BN,num_G, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes,BN,num_G))

        return nn.Sequential(*layers)


    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes) if self.BN else nn.GroupNorm(self.num_G, inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential( conv, bn, self.relu)

    

    def forward(self, x, return_heatmap=False, scale_factor=2):
        num_batch = x.shape[0]
        uv_out = self.UVbranch(x)
        hmp=self.soft( uv_out ) #B, num_class, H,H
        X0=torch.mul(hmp.view(num_batch,self.num_classes,-1),self.Xs)
        X0=torch.sum(X0,dim=-1)
        Y0=torch.mul(hmp.view(num_batch,self.num_classes,-1),self.Ys)
        Y0=torch.sum(Y0,dim=-1)
        X0=torch.unsqueeze(X0,dim=-1)
        Y0=torch.unsqueeze(Y0,dim=-1)
        UV0=torch.cat((X0,Y0),dim=-1)

        aux_attention = self.AttentionEnhBranch(x)
        attentionmap = self.soft( self.Betas *  aux_attention + (1-self.Betas) * uv_out )

        d0= self.DepthBranch(x)

        depht_feat = torch.sum(attentionmap.unsqueeze(2)*d0.unsqueeze(1),dim=(-1,-2)) #B,J,self.depth_dim

        D0 = self.depth_estimator(depht_feat.view(-1,self.depth_dim)).view(num_batch,self.num_classes,1) #B,J,1

        UVD = torch.cat([UV0*scale_factor,D0],dim=-1)

        if return_heatmap:
            return UVD, ( uv_out,aux_attention,attentionmap )

        return UVD


   





def build_EstimationHead(num_blocks, input_dim =256 ,depth_dim =64, num_classes=14, train_spread=False, BN=True,num_G=16):
    return EstimationHead(num_blocks, Bottleneck, input_dim ,depth_dim , num_classes, train_spread, BN, num_G)
