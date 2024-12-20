
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
# torch and lightning imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import SGD, Adam

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from torchmetrics import ConfusionMatrix
from torchmetrics import F1Score, AUROC
from torchvision import transforms
from torch.optim.lr_scheduler import ExponentialLR
import torchmetrics

from torch.optim.lr_scheduler import LinearLR

import numpy as np

import modelos
import metricas
import os
import datetime
import pickle

torch.set_printoptions(precision=3)


   
def write_names(fichero,nombres):
    with open(fichero, 'w') as fp:
        for item in nombres:
            fp.write("%s\n" % item)
    #print(f' Finished writing {fichero}')

class  ConstrainedSegmentMIL(pl.LightningModule):
    def __init__(self,config=None):

        super().__init__()

        if config is None: # dummy constructor # config may be loaded later with load
            return
        
        self.config=config
        self.area_minima=config['train']['min_defect_area']
        self.constrain_weight=config['train']['constrain_weight']  
        self.defect_types=config['model']['defect_types']
        self.num_classes=len (self.defect_types)
        self.num_channels_in=config['model']['num_channels_input'] 
        self.p_dropout=config['train']['p_dropout']
        self.lista_pesos_clase=config['train']['class_weights']
        self.bottom_constrain_weight=config['train']['bottom_constrain_weight']
        self.bin_mask_weight=config['train']['binmask_weight']
        self.min_negative_fraction=config['train']['min_negative_fraction']

        print("SegmentMIL Area Minima Defecto:", self.area_minima)
        print('ConstrainWeight:',self.constrain_weight)
        
        print('SegmentMIL num clases out=',self.num_classes)

           
        model_version=str(config['model']['model_version'])
        if model_version ==  "50":
            self.modelo = modelos.deeplabv3resnet50(num_classes=self.num_classes,num_channels_in=self.num_channels_in, p_dropout=self.p_dropout)
        elif model_version ==  "MobileNet":
            self.modelo = modelos.deeplabv3Mobilenet(num_classes=self.num_classes,num_channels_in=self.num_channels_in, p_dropout=self.p_dropout)            
        else:
            print(f"\n***** Warning. Requested Model {model_version} not implemented. Usando deeplabv3_resnet50")
            self.modelo=modelos.deeplabv3resnet50(num_classes=self.num_classes, p_dropout=self.p_dropout)

        self.epoch_counter=1
        self.estadisticas_labels=None
        self.pos_weights = None
        self.area_minima = config['train']['min_defect_area']
        self.min_negative_fraction = config['train']['min_negative_fraction']
        self.class_names=config['model']['defect_types']
        
    def load(self,config):
        self.config=config
        self.__init__(config)

    def save(self,path):

        directory=self.config['train']['output']['path']
        Path(directory).mkdir(parents=True, exist_ok=True)
        filename=self.config['train']['output']['model_file']
        path=os.path.join(directory,filename)
        salida={'state_dict':self.state_dict(),
        #'normalization_dict':self.normalization_dict, in config
        'training_date': datetime.datetime.now(),
        'final_val_aucs':self.aucs,
        'model_type':'SegmentationMIL',
        'config': self.config
    }

        with open(path, 'wb') as f:
                pickle.dump(salida, f)
        print(f' Finished writing {path}')        
    
    def aggregate_predict(self,logits_pixeles):
        '''
        Dados los logits a nivel de pixel devuelve:
        - top_logits_ordered: top K logits para cada clase de cada elemento del batch. (batch_size, n_classes, K)
        - prob_defecto: probabilidad de defecto de cada clase de cada elemento del batch. (batch_size, n_classes)
        - prob_pixeles: probabilidad de defecto a nivel de pixel. (batch_size, n_classes, H, W)
        '''
        tam = logits_pixeles.shape
        logits_patches_reshaped = torch.reshape(logits_pixeles , (tam[0],tam[1], tam[2]*tam[3] ) )
        
        top_logits_ordered = torch.topk(logits_patches_reshaped,self.area_minima, dim=-1)[0]
        prob_defecto = torch.mean(torch.sigmoid(logits_pixeles),axis=(2,3))
        probs_pixeles = torch.sigmoid(logits_pixeles) 
          
        return top_logits_ordered,prob_defecto,probs_pixeles 
    
    
    def aggregate_train(self,logits_pixeles):
        '''
        La entrada de la función son los logits de la salida del modelo de dimensiones (batch_size, num_classes, H, W)
        Devuelve tres listas:
            -	top_logits_ordered: los top K logits para cada clase de cada vista del batch. (batch_size x num classes x K)
            -	prob_defecto: la media de las probabilidades para cada clase. (batch_size x num classes)
            -	bottom_logits_ordered: los N logits más pequeños, siendo N el número mínimo de pixeles negativos (que no tengan ninguna clase). 
                Con esto se fuerza a que si no hay ninguna clase, la probabilidad en esos pixeles sea bajo, próxima a 0. (batch_size x num classes x N)
        '''
        tam = logits_pixeles.shape
        logits_patches_reshaped = torch.reshape(logits_pixeles , (tam[0],tam[1], tam[2]*tam[3] ) )
        
        top_logits_ordered = torch.topk(logits_patches_reshaped,self.area_minima, dim=-1)[0] 
        ngood_pixels=int(self.min_negative_fraction * logits_pixeles.shape[2]*logits_pixeles.shape[3])
        bottom_logits_ordered = -torch.topk(-logits_patches_reshaped,ngood_pixels, dim=-1)[0] 
        
        prob_defecto = torch.mean(torch.sigmoid(logits_pixeles),axis=(2,3)) 
        
        #self.fraccion_minima= self.area_minima/(logits_pixeles.shape[2]*logits_pixeles.shape[3] )
        
        return top_logits_ordered,prob_defecto,bottom_logits_ordered 
    
                
    def forward_predict(self, X):# Para predicciones
        logits_pixeles=self.modelo(X)
        aggregation = self.aggregate_predict(logits_pixeles) 
        return aggregation  
    
    def forward_train(self, X):# Para training
        logits_pixeles=self.modelo(X)
        aggregation = self.aggregate_train(logits_pixeles) # top_logits_ordered,prob_defecto,bottom_logits_ordered
        return logits_pixeles, aggregation


    def criterion(self, logits_pixeles,aggregation,  labels, bin_masks):#Recibe batch_size x (numclases)
        '''
        labels: lista con batch_size elementos. Cada elemento tiene tantos elementos como etiquetas tenga la vista
        logits_pixeles: logits de dims (batch_size,num_classes, H, W) 
        bin_masks: máscaras binarias -> dims (batch_size, num_classes, H, W), si no existe la máscara, entonces será None

        Devuelve cuatro funciones de perdidas:
        - loss_cross_entropy: BCE loss. Se calcula entre los top k logits y las etiquetas si no hay máscara.
                                Si hay máscara, se calcula entre los logits_pixeles y la máscara binaria.

        - bottom_loss: media del Binary Cross Entropy Loss entre los bottom_logits y cero, quedando (batch_size x 1 x 1)

        - loss_constrain_max_area: A partir de las probabilidades de defecto, se obitene la probabilidad de que sea bueno. 
            Y se calcula la media del resultado de calcular: ReLU(min_negative_fraction – prob_bueno)^2. (batch_size x 1 x 1).

        - global_loss: suma ponderada de las tres losses anteriores

        '''

        top_logits=aggregation[0] # Top K logits para cada clase (batch_size x num_classes x K)
        prob_defecto=aggregation[1] # Media de las probs para cada clase (batch_size x num_classes)
        bottom_logits=aggregation[2] # N logits más pequeños. (batch_size, num_classes, N)
        
        #target_fracciones_minimas=labels * self.fraccion_minima
        bcelogitsloss=nn.BCEWithLogitsLoss(reduction='mean')
        
        ntipos_defecto=top_logits.shape[1]
        
        batch_size=labels.shape[0]

        losses=[]
        bottom_losses=[]
      
        for k in range(ntipos_defecto): # para cada clase
            listapesos=[]
            class_mean_losses=[]
            for b in range(batch_size): # para cada elemento b del batch 
                bm_b=bin_masks[b] # Máscaras binaria del elemento b del batch
                labels_b=labels[b] # Etiquetas del elemento b del batch
               
                logits_pixeles_b=logits_pixeles[b]
                top_logits_b=top_logits[b]
                
                logits=top_logits_b[k,:] # los toppixeles de la clase k del elemento b del batch
                target=labels_b[k] # Este tipo de defecto en esta vista
                if target.isnan().any():
                    continue
            
                if bm_b[k] is  None: # No hay mascara binaria para la clase k de la vista b
                                             
                    batch_element_class_loss = bcelogitsloss(logits,torch.full_like(logits,target))#create tensor same size as logits with target values
                    listapesos.append(1.0)
                    bottom_loss = bcelogitsloss(bottom_logits[b,k,:],torch.zeros_like(bottom_logits[b,k,:]))
                    bottom_losses.append(bottom_loss)                
                else: # Hay máscara binaria
                    batch_element_class_loss = self.binmask_weight*bcelogitsloss(logits_pixeles_b[k],bm_b[k])
                    listapesos.append(self.binmask_weight) #asignar peso > 1 ya que hay muchas menos imagenes con mascara
            
                class_mean_losses.append(batch_element_class_loss) # se generará una lista de batch_size elementos para cada clase
            if len(class_mean_losses)==0:
                print(" *** No hay elementos en la clase ",self.class_names[k])
                losses.append(torch.tensor(0.0,device=self.device))
            else:
                class_losses=torch.stack(class_mean_losses)#batch_size
                listapesostensor=torch.tensor(listapesos)
                loss_cross_entropyc=torch.sum(class_losses)/listapesostensor.sum() #media ponderada de los losses de un batch de una clase
                losses.append(loss_cross_entropyc) # será una lista de n_classes elementos


        pesos_clase=torch.tensor(self.lista_pesos_clase,device=self.device)
        pesos_clase/=pesos_clase.sum()
        loss_cross_entropy = torch.sum(torch.stack(losses)*pesos_clase) # suma ponderada de los losses por clase
     
        
        bottom_loss=torch.mean(torch.tensor(bottom_losses))
        if loss_cross_entropy.isnan().any():
            print("Loss cross entropy is nan", loss_cross_entropy)
            print('class_losses stack',class_losses)
            print('targets',labels)
            self.logger.experiment.finish()
            exit()
        prob_good=1-prob_defecto # probabilidad de que no esté el defecto (batch_size, num_classes)
        loss_constrain_max_area= torch.relu(self.min_negative_fraction-prob_good)**2
        loss_constrain_max_area = torch.mean(loss_constrain_max_area)
        
        
        global_loss=loss_cross_entropy + self.constrain_weight * loss_constrain_max_area +self.bottom_constrain_weight*bottom_loss
        #print(f'Losses: BCE={loss_cross_entropy}, constrain={loss_constrain_max_area}, global={global_loss} ')
        
        return global_loss, loss_cross_entropy, self.constrain_weight *loss_constrain_max_area, bottom_loss


    def configure_optimizers(self):
        
        optimizer=self.config['train']['optimizer']
        gamma_param=self.config['train']['gamma_param']
        warmup_iter=self.config['train']['warmup']
        num_epochs=self.config['train']['epochs']
        weight_decay=self.config['train']['weights_decay']
        lr=self.config['train']['learning_rate']
        parameters = list(filter(lambda x: x.requires_grad, self.parameters()))
        if optimizer.lower() == 'sgd':
            optimizer = SGD(parameters, lr=lr, weight_decay=weight_decay)
        elif optimizer.lower() == 'adam':
            optimizer = Adam(parameters, lr=lr, weight_decay=weight_decay)
        else:
            print(f'**** WARNING : Optimizer configured to {optimizer}. Falling back to SGD')   
            optimizer = SGD(parameters, lr=lr, weight_decay=self.weight_decay)
                
        gamma=gamma_param**(1/num_epochs)
        if warmup_iter > 0:           
            warmup_lr_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_iter)
            schedulers = [warmup_lr_scheduler, ExponentialLR(optimizer, gamma=gamma) ]
        else:
           schedulers = [ ExponentialLR(optimizer, gamma=0.99) ] 
        return [optimizer],schedulers


    def training_step(self, batch, batch_idx):
        #print("New training step")
        images = batch['images']
        labels = batch['labels']
        folders=batch['imags_folder']
        bin_masks=batch['bin_masks']
        if 'casos' in batch:
            casos=batch['casos']
        else:
            casos=batch['view_ids']
        
        # if labels.isnan().any():
        #     print('Labels contain nans in train step. Stopping')
        #     print('Labels',labels )

  
        logits_pixels, aggregation = self.forward_train(images)
        
        global_loss,loss_cross_entropy,loss_constrain_max_area,bottom_loss = self.criterion(logits_pixels, aggregation,  labels,bin_masks)
      

        if self.estadisticas_labels is None:
            self.estadisticas_labels = labels   
        else:
            self.estadisticas_labels=torch.concat((self.estadisticas_labels,labels),dim=0)

        if global_loss.isnan():
            if images.isnan().any():
                print('Images contain nans in train step. Stopping')
                self.logger.experiment.finish()
            else:
                print('Images do not contain nans in train step.')
                
            #print('Loss is nan in train step. Stopping')
           # self.logger.experiment.finish()
            #print('Labels',labels )
            #print('Folders',folders)
            #print('Casos',casos)
            #exit()
        # perform logging
    
        log_dict={'train_loss':global_loss,'train_BCEloss':loss_cross_entropy,'train_constrain_loss':loss_constrain_max_area}
        if self.bottom_constrain_weight > 0:
            log_dict['train_bottom_loss']=bottom_loss

        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return global_loss



    def validation_step(self, batch, batch_idx):
        images = batch['images']
        labels = batch['labels']
        folders=batch['imags_folder']
        bin_masks=batch['bin_masks']
        if 'casos' in batch:
            casos=batch['casos']
        else:
            casos=batch['view_ids']
        
        casos=[ os.path.join(c[0],c[1]) for c in zip(folders,casos)]
               
        logits_pixels, aggregation = self.forward_train(images)
       

        global_loss,loss_cross_entropy,loss_constrain_max_area,bottom_loss = self.criterion(logits_pixels, aggregation,  labels,bin_masks)

        if global_loss.isnan():
            print('Loss is nan in val step. Stopping')
            self.logger.experiment.finish()
            print('Labels',labels )
            print('Folders',folders)
            print('Casos',casos)
            exit()
   
        log_dict={'val_loss':global_loss,'val_BCEloss':loss_cross_entropy,
                  'val_constrain_loss':loss_constrain_max_area}
        if self.bottom_constrain_weight > 0:
            log_dict['val_bottom_loss']=bottom_loss

        self.log_dict(log_dict , on_step=False, on_epoch=True, prog_bar=True, logger=True)




    def predict_step(self, batch, batch_idx, dataloader_idx=0):    
        pass

    def on_validation_epoch_end(self, ) -> None:        
        self.epoch_counter += 1 
        return    



    def on_training_epoch_end(self) -> None:
        pass
            

