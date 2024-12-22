
import warnings
from pathlib import Path
from argparse import ArgumentParser
import argparse
import os

import yaml
import json
import pickle

warnings.filterwarnings('ignore')


import sys
current_file_dir = os.path.dirname(os.path.abspath(__file__))

# Define the relative path to the folder you want to add

common_folder = os.path.join(current_file_dir, "../common")
sys.path.append(common_folder)
sys.path.append(current_file_dir)
# print("PATHS:",sys.path)

# torch and lightning imports
import torch
torch.set_float32_matmul_precision('medium')
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_lightning.callbacks import  ModelCheckpoint
import multiprocessing
import base64
from PIL import Image
import io

from tqdm import tqdm
from pl_datamodule import DataModule, my_collate_fn

import pl_datamodule
import torch.nn.functional as F
#from dataset import ViewsDataSet
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from metricas import calculate_auc_multilabel
from pl_constrained_model import ConstrainedSegmentMIL
import m_dataLoad_json

from datetime import datetime


#from m_dataLoad_json import write_list

# Función para serializar tensores
def tensor_to_serializable(obj):
    if isinstance(obj, torch.Tensor):  # Si es un tensor de PyTorch
        return obj.tolist()  # Convierte a lista
    raise TypeError(f"Objeto de tipo {type(obj).__name__} no es serializable.")


if __name__ == "__main__":
    # parser = ArgumentParser()

    # parser.add_argument("--config", default = "configs/train.yaml", help="""YAML config file""")

    # args = parser.parse_args()

    config_file = sys.argv[1]
    with open(config_file,'r') as f:
        config=yaml.load(f,Loader=yaml.FullLoader)
    
    
       # Comprobar si hay GPU
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

    train_dataplaces = config['data']['train']
    val_dataplaces = config['data']['val']
    terminaciones=config['data']['suffixes']
    root_folder=config['data']['root_folder']
    
    # crop_size=config['data']['crop_size']
    delimiter=config['data']['delimiter']
    # batch_size=config['train']['batch_size']
    # in_memory=config['train']['in_memory']
    maxvalues=config['data']['maxvalues']
    crop_size=config['data']['crop_size']
    tipos_defecto=config['model']['defect_types']
    
    
    model_dir=config['evaluate']['model_dir']
    model_file=config['evaluate']['model_file']
  
    
    #multilabel=config['model']['multilabel']

    out_dir=config['evaluate']['report_dir']

    
    if not os.path.exists(out_dir):
        print(f'No existe el directorio para report : {out_dir}. Se crea')
        Path( out_dir ).mkdir( parents=True, exist_ok=True )

   
    train_predictions=config['evaluate']['train_predictions']
    val_predictions=config['evaluate']['val_predictions']
    aucs_jsonfile=config['evaluate']['aucs']

    model_path=os.path.join(model_dir,model_file)
    
    model=ConstrainedSegmentMIL()
    model.load(model_path)
    
    training_date=str(model.training_date)
    
    print("\n\n====================================================================")
    print("  ******************** TRAINING SET *************************")
    print("====================================================================")        
  
    preds_train=[]
    targets_train=[]
    #ids_train=[]
    json_data=[]

    print('root_folder:',root_folder)
    print('train_dataplaces num_directories',len(train_dataplaces)) 
    dataset,tipos_defecto=m_dataLoad_json.genera_ds_jsons_multilabel(root_folder,  train_dataplaces, sufijos=terminaciones,maxValues=maxvalues, crop_size=crop_size,
                                                                            defect_types=tipos_defecto,splitname_delimiter=delimiter,multilabel=True, in_memory=False)
    class_results=[]


    for caso in tqdm(dataset):
        targets=caso['labels']
        fruit_id=caso['fruit_id']
        view_id=caso['view_id']
        imags_folder=caso['imag_folder']
        view_id=os.path.join(imags_folder,view_id)
        results=model.predict(view_id,device,include_images=False,remove_suffix=False)
        result= results[0]
       
        preds_train.append(result['view_probs_tensor'])
        targets_train.append(targets.unsqueeze(dim=0))
        targets_dict={}
        
        for nclase in range(len(tipos_defecto)):
            targets_dict[tipos_defecto[nclase]]=int(targets[nclase].item())

        json_data.append({'filename': result['imgname'],
                            'scores':result['view_probs'],
                            'ground_truth': targets_dict})
                        

    data_train={'train_results':json_data,
                'evaluation_date': str(datetime.now()),
                'training_date':training_date,
                'model_file':model_path}
    out_json_trainfile=os.path.join(out_dir,train_predictions)

    #print("Dataval:",data_val)
    print("Writing ",out_json_trainfile)
    with open(out_json_trainfile, 'w') as f:
        json.dump(data_train, f, indent=3)
    

    preds_train = torch.stack(preds_train)
    targets_train = torch.concat(targets_train)
    aucs_train=calculate_auc_multilabel(preds_train,targets_train,tipos_defecto)   

    preds_val=[]
    targets_val=[]
    #ids_train=[]
    json_data=[]
    print("\n\n====================================================================")
    print("  ******************** VALIDATION SET *************************")
    print("====================================================================")
    print('root_folder:',root_folder)
    print('val_dataplaces num_directories',len(val_dataplaces)) 
    dataset,tipos_defecto=m_dataLoad_json.genera_ds_jsons_multilabel(root_folder,  val_dataplaces, sufijos=terminaciones,maxValues=maxvalues, crop_size=crop_size,
                                                                            defect_types=tipos_defecto,splitname_delimiter=delimiter,multilabel=True, in_memory=False)
    class_results=[]
    for caso in tqdm(dataset):
        #image=caso['image']
        targets=caso['labels']
        fruit_id=caso['fruit_id']
        view_id=caso['view_id']
        imags_folder=caso['imag_folder']
        view_id=os.path.join(imags_folder,view_id)
        results=model.predict(view_id,device,include_images=False,remove_suffix=False)
        result= results[0]
        #print("Result:",result) 
        #results=model.predict(image,device)
        #print(result['view_probs_tensor'])
        preds_val.append(result['view_probs_tensor'])
        targets_val.append(targets.unsqueeze(dim=0))
        targets_dict={}
        
        for nclase in range(len(tipos_defecto)):
            targets_dict[tipos_defecto[nclase]]=int(targets[nclase].item())

        json_data.append({'filename': result['imgname'],
                            'scores':result['view_probs'],
                            'ground_truth': targets_dict})
                        


    data_val={'val_results':json_data,
              'evaluation_date': str(datetime.now()),
              'training_date':training_date,
              'model_file':model_path}
    
    out_json_valfile=os.path.join(out_dir,val_predictions)

    #print("Dataval:",data_val)
    print("Writing ",out_json_valfile)
    with open(out_json_valfile, 'w') as f:
        json.dump(data_val, f, indent=3)
         
    

    # preds_train = torch.concat(preds_train)
    # targets_train = torch.concat(targets_train)

    print("\n====================================================================")
    print("  ******************** AUCs *************************")
    print("====================================================================")

    preds_val = torch.stack(preds_val)
    targets_val = torch.concat(targets_val)    
    aucs_val=calculate_auc_multilabel(preds_val,targets_val,tipos_defecto)

    
    aucdata={'Train AUCs':aucs_train,
             'Val AUCs': aucs_val,
             'evaluation_date': str(datetime.now()),
             'training_date':training_date,
             'model_file':model_path}

    aucs_file=os.path.join(out_dir,aucs_jsonfile)
    print("Writing ",aucs_file)
    with open(aucs_file, 'w') as f:
        json.dump(aucdata, f, indent=3)
    
     


    
    