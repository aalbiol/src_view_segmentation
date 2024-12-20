

import os

import torch
import torch.onnx
from torch.autograd import Variable
from torchvision import models

from argparse import ArgumentParser
import argparse
import pytorch_lightning as pl
import json


from modelos import resnet50MIL, deeplabv3resnet50


#from torchsummary import summary


from torchinfo import summary

def main():
    parser = ArgumentParser()

 
    parser.add_argument( "--model_version", default=None, help=""" Red a la que hacer el summary""")
    parser.add_argument( "--image_size", help="Image target size.", type=int, default=110)
    parser.add_argument("--batch_size", help="Batch size", type = int, default=10)
    parser.add_argument("--multilabel", action=argparse.BooleanOptionalAction)
    parser.set_defaults(multilabel=True)

    args = parser.parse_args()

    with open('clases.json') as json_file:
        clases = json.load(json_file)
    
    with open('normalizacion.json') as json_file:
        normalizacion = json.load(json_file)    

    nombres_clases=clases['clases']
    print('nombres de clases: ', nombres_clases)

    channel_list=[0,1,2,3]#RGB-NIR
    #model = FruitMILClassifier(                          
    #                    num_channels_in=4,
     #                   multilabel=True,
      #                  model_version=50,                     
       #                 class_names=nombres_clases
        #            )
    model = deeplabv3resnet50(num_classes=11, p_dropout=0.5)
    #model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights, weights_backbone=models.ResNet50_Weights.IMAGENET1K_V1)
    
    #model = resnet50MIL(num_classes=11, p_dropout=0.5)              
    input_size=(1,3,240,240)
    summary(model, input_size=(1,3,224,224),col_names = ('input_size', 'output_size', 'num_params','params_percent', 'kernel_size','mult_adds','trainable'))
    #repr(model)
   
    #print(summary(model, (3, 110, 110), batch_dim = -1, col_names = ('input_size', 'output_size', 'num_params', 'kernel_size','mult_adds'), verbose = 1))
   


if __name__ == "__main__":
    main()



