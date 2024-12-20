import warnings

from pathlib import Path
from argparse import ArgumentParser
import argparse
import os
warnings.filterwarnings('ignore')

import sys
current_file_dir = os.path.dirname(os.path.abspath(__file__))

# Define the relative path to the folder you want to add

common_folder = os.path.join(current_file_dir, "../common")
sys.path.append(common_folder)
sys.path.append(current_file_dir)

# torch and lightning imports
import torch
torch.set_float32_matmul_precision('medium')
torch.multiprocessing.set_sharing_strategy('file_system')
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import  ModelCheckpoint
from m_finetuning import FeatureExtractorFreezeUnfreeze
import json

from pl_datamodule import DataModule
from pl_constrained_model import ConstrainedSegmentMIL

#from imgcallback import ImageLogger

import pathlib

import yaml

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print('Usage: train.py <config_file>')
        sys.exit(1)

    config_file = sys.argv[1]

    with open(config_file,'r') as f:
        config=yaml.load(f,Loader=yaml.FullLoader)
       
    # Check GPU availability
    cuda_available=torch.cuda.is_available()
    if cuda_available:
        cuda_count=torch.cuda.device_count()    
        cuda_device=torch.cuda.current_device()
        tarjeta_cuda=torch.cuda.device(cuda_device)
        tarjeta_cuda_device_name=torch.cuda.get_device_name(cuda_device)
        print(f'Cuda Disponible. Num Tarjetas: {cuda_count}. Tarjeta activa:{tarjeta_cuda_device_name}\n\n')
        device='cuda'
        gpu=1
    else:
        device='cpu'
        gpu=0
   
    datamodule =  DataModule(config)
        
    print('Categories:',datamodule.categories)

    means_norm=datamodule.mean_norm.tolist()
    stds_norm=datamodule.stds_norm.tolist()
    size_rect=str(datamodule.size_rect)
 
    dict_norm={'means_norm': means_norm, 'stds_norm': stds_norm }
    
    config['data']['dict_norm']=dict_norm    

    model = ConstrainedSegmentMIL(config) # It loads initial weights if specified in config file

    project=config['train']['wandb_project']
    logname=config['train']['logname']
    miwandb= WandbLogger(name=logname, project=project,config=config)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks4=[lr_monitor, 
                FeatureExtractorFreezeUnfreeze(unfreeze_at_epoch=config['unfreeze_epoch'],initial_denom_lr=1)]
    num_epochs=config['train']['epochs']
    trainer_args = {'max_epochs': num_epochs, 'logger' : miwandb}

    trainer = pl.Trainer(callbacks=callbacks4 , **trainer_args) 
    
    trainer.fit(model, datamodule=datamodule)
    model.save() # Save model in the path specified in the config file

    
    
    
