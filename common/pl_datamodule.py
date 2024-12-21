import warnings
warnings.filterwarnings('ignore')

# torch and lightning imports
import torch
import pytorch_lightning as pl

from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pycimg
import multiprocessing
from typing import Tuple,Any

import os
from PIL import Image
import sampler
import cv2
import pandas as pd

import m_dataLoad_json

from transformaciones import Aumentador_Imagenes_y_Mascaras


class FileNamesDataSet(Dataset):
    '''
    Clase para suministrar archivos de una lista cuando no hay anotacion
    Solo se devuelve la imagen y el fruitid y el viewid.
    Por compatibilidad se devuelve un target = None
    '''
    def __init__(self,root_folder=None, filenames_list=None ,transform=None, 
                 field_delimiter='-',max_value = 255, *args, **kwargs):
        super().__init__(*args,  **kwargs)
        
        self.root_folder=root_folder
        self.filenames_list=list(filenames_list)
        self.transform = transform
        self.field_delimiter=field_delimiter
        self.max_value = max_value
        
              
    def __getitem__(self, index: int) -> Tuple[Any, Any]:              
        view_id=m_dataLoad_json.view_id(self.filenames_list[index])
        fruit_id=m_dataLoad_json.fruit_id(self.filenames_list[index],delimiter=self.field_delimiter)
        
        img_fullname=(self.filenames_list[index] if self.root_folder is None else os.path.join(self.root_folder,self.filenames_list[index]))
        imagen= m_dataLoad_json.lee_png16(img_fullname,self.max_value)
           
        if self.transform is not None:                
            imagen2 = self.transform(imagen)
        else:
            imagen2=imagen
        target=None     # Este data set no está etiquetado               
        return imagen2, target, view_id,fruit_id 
              
    def __len__(self) -> int:
        return len(self.filenames_list)
 
 
 
class ViewsDataSet(Dataset):
    def __init__(self,dataset=None ,transform=None, use_masks=None,*args, **kwargs):
        super().__init__(*args,  **kwargs)
        
        self.dataset=dataset   
        self.transform = transform
        self.use_masks=use_masks 
              
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        caso=self.dataset[index]
        target =caso['labels'].float()
        view_id=caso['view_id']
        fruit_id=caso['fruit_id']
        imagen=caso['image']
        imags_folder=caso['imag_folder']
        bin_masks=caso['bin_masks']
        if imagen is None: #Cuando no está en memoria        
            sufijos=caso['sufijos']
            maxValues=caso['maxValues']
            crop_size=caso['crop_size']
            d=caso['dict_json']
            tipos_defecto=caso['tipos_defecto']
            #print("Reading ", view_id)
            imagen=m_dataLoad_json.lee_vista(imags_folder,view_id,sufijos,maxValues,crop_size=crop_size)
            bin_masks=m_dataLoad_json.lee_mascaras(imags_folder,view_id,'_RGB_mask.png',255,
                                                   crop_size,d,tipos_defecto,self.use_masks)          
        if self.transform is not None:              
            imagen2,bin_masks2 = self.transform(imagen,bin_masks)  
        else:
            imagen2=imagen
            bin_masks2=bin_masks
                               
        return imagen2,target,view_id,fruit_id,imags_folder,bin_masks2 # imagen, lista_de_etiquetas, (ruta_al_archivo, vista_id) 
       
    def __len__(self) -> int:
        return len(self.dataset)
    

def matriz_etiquetas(dataset):
    lista=[]
    for caso in dataset:
        labels=caso['labels']
        lista.append(labels)
    matriz=torch.stack(lista)

    return matriz

def casos_mask(dataset,clase,use_masks):
    lista=[]
    print("casos_mask: clase=",clase)
    for caso in dataset:
        view_id=caso['view_id']
        res=0
        if use_masks:
            if 'masks' in caso['dict_json']:
                masks=caso['dict_json']['masks']
                
                masks={k.lower():v for k,v in masks.items()}
                #print("masks",masks)
                if clase in masks:
                    
                    valor=masks[clase.lower()]
                    if len(valor)>0:
                        res=1
                        #print ("Añadiendo a la lista de imagenes con mascara ", view_id)
        lista.append(res)
    return torch.tensor(lista)

def view_ids(dataset):
    lista=[]
    for caso in dataset:
        v_id=caso['view_id']
        lista.append(v_id)

    return lista
        
    
 
def my_collate_fn(data): # Crear batch a partir de lista de casos
    '''
    images: tensor de batch_size x num_channels_in x height x width
    labels: tensor de batch_size x num_classes
    view_ids: lista de batch_size elementos 
    fruit_ids: lista de batch_size elementos 
    bin_masks: lista de batch_size elementos
    '''
    images = [d[0] for d in data]
    # tams=[im.size()  for im in images]
    # print("Tamaños de las imagenes en el batch", tams)
    images = torch.stack(images, dim = 0) # tendra dimensiones numvistastotalbatch, 3,250,250

    labels = [d[1] for d in data]
    if labels[0] is not None:
        labels = torch.stack(labels,dim=0)
    #labels = torch.tensor(labels).long()
    #labels es una lista con tantos elementos como batch_size
    # Cada elemento
    view_ids = [d[2] for d in data]
    fruit_ids = [d[3] for d in data]  
    imags_folder = [d[4] for d in data]  
    bin_masks = [d[5] for d in data]

    return { 
        'images': images, 
        'labels': labels,
        'view_ids': view_ids,
        'fruit_ids': fruit_ids,
        'imags_folder':imags_folder,
        'bin_masks': bin_masks #del batch
    }
 


class DataModule(pl.LightningDataModule):
    def __init__(self, config=None):
        assert config is not None, "Config in DataModule is None"
        super().__init__()            
        self.config=config
        
        self.trainset=None
        self.valset=None
        self.predset=None
        train_dataplaces=config['data']['train']
        val_dataplaces=config['data']['val']
        
        if train_dataplaces is not None:
            print("\nGenerating training dataset...")
            self.trainset,self.tipos_defecto=m_dataLoad_json.genera_ds_jsons_multilabel(self.config['data']['root_folder'], 
                                                                                                    dataplaces=train_dataplaces, 
                                                                                                    sufijos=config['data']['suffixes'],
                                                                                                    maxValues=config['data']['maxvalues'],
                                                                                                    crop_size=config['data']['crop_size'], 
                                                                                                    training_size=config['data']['training_size'],
                                                                                                    defect_types=config['model']['defect_types'],
                                                                                                    multilabel=True,
                                                                                                    splitname_delimiter=config['data']['delimiter'],
                                                                                                    in_memory=config['train']['in_memory'],
                                                                                                    use_masks=config['data']['use_segmentation_masks'])

        if val_dataplaces is not None:
            print("\nGenerating validation dataset...")
            self.valset,self.tipos_defecto=m_dataLoad_json.genera_ds_jsons_multilabel(self.config['data']['root_folder'], 
                                                                                                    dataplaces=val_dataplaces, 
                                                                                                    sufijos=config['data']['suffixes'],
                                                                                                    maxValues=config['data']['maxvalues'],
                                                                                                    crop_size=config['data']['crop_size'], 
                                                                                                    training_size=config['data']['training_size'],
                                                                                                    defect_types=config['model']['defect_types'],
                                                                                                    multilabel=True,
                                                                                                    splitname_delimiter=config['data']['delimiter'],
                                                                                                    in_memory=config['train']['in_memory'],
                                                                                                    use_masks=config['data']['use_segmentation_masks'])
          

        
        print(f"Trainset: {len(self.trainset)}")
        print(f"Valset: {len(self.valset)}")    
        assert len(self.trainset)>0, "Trainset is empty"
        assert len(self.valset)>0, "Valset is empty"

        print('Computing means and stds...')

        self.categories=self.tipos_defecto
        
        self.means_norm, self.stds_norm=m_dataLoad_json.calcula_media_y_stds(self.trainset)
        print(f'Means: {self.means_norm}')
        print(f'Stds: {self.stds_norm}')

        training_size=config['data']['training_size']   

        augmentation=config['train']['augmentation']

        affine=augmentation['affine']

        transform_geometry_train= transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(augmentation['random_rotation']),
            transforms.RandomAffine(degrees=affine['degrees'], shear=affine['shear'], scale=affine['scale']),
            transforms.RandomRotation(augmentation['random_rotation']),
            transforms.Resize(training_size),
            ])
        
        transform_geometry_val= transforms.Compose([
            transforms.Resize(training_size),
            ])
        

        transform_intensity_train= transforms.Compose([
            transforms.ColorJitter(brightness=augmentation['brightness'], hue=augmentation['hue'],contrast=augmentation['contrast'],saturation=augmentation['saturation']),            
            ])
        transform_normalize=transforms.Compose([transforms.Normalize(self.means_norm, self.stds_norm),
                                                ])

        transform_train=Aumentador_Imagenes_y_Mascaras(transform_geometry_train,
                                                       transform_intensity_train,transform_normalize)

        transform_val = Aumentador_Imagenes_y_Mascaras(transform_geometry_val,
                                                       None,transform_normalize)
        
        
        self.train_dataset=None
        self.val_dataset=None
        self.pred_dataset=None
        use_masks=config['data']['use_segmentation_masks']
        if self.trainset is not None:
            self.train_dataset = ViewsDataSet(dataset=self.trainset, transform = transform_train,use_masks=use_masks)    
        if self.valset is not None:                   
            self.val_dataset = ViewsDataSet(dataset=self.valset, transform = transform_val, use_masks=use_masks) 
            
                          
        if self.trainset is not None: # Si estamos en prediccion no lo hago
            self.matriz_casos_train=matriz_etiquetas(self.trainset)
            self.matriz_casos_val=matriz_etiquetas(self.valset)
            self.view_ids_train=view_ids(self.trainset)
            self.view_ids_val=view_ids(self.valset)

        print(">>>>>>>>>>>>>>>>>>")
        if self.train_dataset is not None:
            print(f"len total trainset =   {len(self.train_dataset )}")

        if self.val_dataset is not None:
            print(f"len total valset =   {len(self.val_dataset )}")
    
        
    def get_len_trainset(self):
        return len(self.train_dataset)
    




    def prepare_data(self):
        pass

    def setup(self, stage=None):
        return None

    def train_dataloader(self):
        self.batch_size=self.config['train']['batch_size']
        print("batch_size in Dataloader train", self.batch_size)
        use_masks=self.config['data']['use_segmentation_masks']
        self.num_workers=self.config['train']['num_workers']
        print("Numworkers leido de config", self.num_workers)
        if self.num_workers < 0:
            self.num_workers=multiprocessing.cpu_count()-2
            if self.num_workers < 0:
                self.num_workers=1
        print("num_workers in train_dataloader", self.num_workers)
        
        # if self.multilabel == False:
        #     misampler=sampler.Balanced_BatchSampler(self.trainset)
        # else:
        misampler=sampler.Balanced_BatchSamplerMultiLabel(self.trainset, 
                                                                  self.tipos_defecto,use_masks=use_masks)

        return DataLoader(self.train_dataset, batch_size=self.batch_size,sampler=misampler, num_workers=self.num_workers, collate_fn=my_collate_fn)
    
    def val_dataloader(self):
        self.batch_size=self.config['train']['batch_size']
        print("batch_size in Dataloader train", self.batch_size)        
        self.num_workers=self.config['train']['num_workers']
        print("Numworkers leido de config", self.num_workers)
        if self.num_workers < 0:
            self.num_workers=multiprocessing.cpu_count()-2
            if self.num_workers < 0:
                self.num_workers=1
        print("num_workers in train_dataloader", self.num_workers)
        print("num_workers in train_dataloader", self.num_workers)        
        print("batch_size in Dataloader val", self.batch_size)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=False,shuffle=False, collate_fn=my_collate_fn)

    
    def predict_dataloader(self):
        pass
        
