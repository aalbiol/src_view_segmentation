import torch.nn as nn
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, deeplabv3_resnet50, DeepLabV3
import torchvision.models as models
from collections import OrderedDict
import torch
from torchvision.models import ResNet


class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.log_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.log_weight)

    def forward(self, input):
        return nn.functional.linear(input, self.log_weight.exp())
    
class DeepLabV3Resnet50(nn.Module):
    def __init__(self, model, num_channels_in, num_classes, p_dropout=0.5):
        
        super(DeepLabV3Resnet50, self).__init__()
        """self.features = nn.Sequential( OrderedDict([
            ('conv1', model.backbone.conv1),
            ('bn1',model.backbone.bn1),
            ('relu', model.backbone.relu),
            ('maxpool',model.backbone.maxpool),
            ('layer1',model.backbone.layer1),
            ('layer2',model.backbone.layer2),
            ('layer3',model.backbone.layer3),
            ('layer4',model.backbone.layer4),
            ]))"""
        #num_channels_in=4
        print('DeepLab Num channels_in:',num_channels_in)
        self.num_channels_in=num_channels_in
        self.num_classes=num_classes
        self.conditioner=None
        if self.num_channels_in !=3:
            self.conditioner=nn.Conv2d(self.num_channels_in, 3, kernel_size=1, stride=1, padding=0, bias=False)

        self.features=model.backbone
        
        #self.classifier=model.classifier
        
        self.model=model
        self.model.classifier[4]= torch.nn.Conv2d(256, self.num_classes, kernel_size=(1, 1), stride=(1, 1)) #ultima capa convolucional del modelo
        
        self.model.aux_classifier[4]=torch.nn.Conv2d(256, self.num_classes, kernel_size=(1, 1), stride=(1, 1)) #opcional, no se utiliza
        
        
    
    def forward(self,x):
        #print('model out',self.model(x)['out'].shape)
        #print('self num channels in', self.num_channels_in)
        if self.num_channels_in !=3:
            y= self.conditioner(x)
        else:
            y=x
       
        return self.model(y)['out'] #logits (batch_size, n_classes, H, W)
    
    
    
class DeepLabV3Resnet50Softmax(nn.Module):
    def __init__(self, model, num_classes, p_dropout=0.5):
        super(DeepLabV3Resnet50Softmax, self).__init__()
        """self.features = nn.Sequential( OrderedDict([
            ('conv1', model.backbone.conv1),
            ('bn1',model.backbone.bn1),
            ('relu', model.backbone.relu),
            ('maxpool',model.backbone.maxpool),
            ('layer1',model.backbone.layer1),
            ('layer2',model.backbone.layer2),
            ('layer3',model.backbone.layer3),
            ('layer4',model.backbone.layer4),
            ]))"""
        
        self.features=model.backbone
        
        self.classifier=model.classifier
        self.aux_classifier=model.aux_classifier
        self.model=model
        self.classifier[4]= torch.nn.Conv2d(256, num_classes+1, kernel_size=(1, 1), stride=(1, 1)) 
        self.aux_classifier[4]=torch.nn.Conv2d(256, num_classes+1, kernel_size=(1, 1), stride=(1, 1))
        
        
    
    def forward(self,x):
        # El logit del fondo es el último
        logits=self.model(x)['out']  
        logits_excluyentes=torch.log_softmax(logits,dim=1)
        logits_defectos_pixeles=logits_excluyentes[:,:-1,:,:]
        # Se devuelven tantos logits como defectos. El del fondo se omite
        return logits_defectos_pixeles 
        
    

def deeplabv3resnet50(num_classes, num_channels_in,p_dropout=0.5):
    model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights, weights_backbone=models.ResNet50_Weights.IMAGENET1K_V1,aux_loss=True, pointrend=True)
    print('MODEL NUM CHHANELS IN',num_channels_in)
    return DeepLabV3Resnet50(model, num_classes=num_classes, num_channels_in=num_channels_in,p_dropout=p_dropout)

def deeplabv3resnet50softmax(num_classes, p_dropout=0.5):
    model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights, weights_backbone=models.ResNet50_Weights.IMAGENET1K_V1,aux_loss=True, pointrend=True)
    return DeepLabV3Resnet50Softmax(model, num_classes, p_dropout=p_dropout)
    

