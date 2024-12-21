
import torch


#import dataLoad

import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.data  import Sampler, BatchSampler, SubsetRandomSampler

import pl_datamodule
#########################################
## Funciones para manejar datasets multilabel posiblemente bastante desbalanceados
########################################

# def get_matriz_casos(dataset):
#     ''' matriz de casos de un CImgFruitsViewsDataSet
#     '''
#     matriz_casos=[]
#     for f in dataset.targets:
#         matriz_casos.append(f)
#     matriz_casos=torch.stack(matriz_casos)       
#     return matriz_casos

def get_class_items(dataset,clase):
    ''' 
    dataset: CImgFruitsViewsDataSet
    if clase >=0 lista de elementos en los que clase esta a 1
    if clase < 0 lista de elementos en los que todas las clases estan a 0
    '''
    matriz_casos=pl_datamodule.matriz_etiquetas(dataset)
    
    if clase >=0:
        indices=torch.argwhere(matriz_casos[:,clase]>=0.5)
    else: # Todas las labels nulas
        suma=torch.nansum(matriz_casos,axis=1)
        indices=torch.argwhere(suma==0)   #tensor
    
    indices=[x.item() for x in indices] # Lista de enteros
    
    
    return indices

def get_class_items_with_mask(dataset,clase,use_masks):
    ''' 
    dataset: TomatoViewsDataSet
    clase: indice de la clase
    use_masks: if true -> casos_mask devuleve lista de nbatch elementos con valor 0 o 1:
                    0 si no hay mascara binaria
                    1 si la hay
                if false -> devuelve lista de nbatch ceros.
    Devuelve lista de indices de los casos donde hay máscara.
    '''
    
    print("Sampler use_masks",use_masks)
    casos_mascara=pl_datamodule.casos_mask(dataset,clase,use_masks) #Lista de nbatch elementos: 0 si no hay mascara binaria para el elemento de la clase, 1 si la hay
    
    if casos_mascara.sum() is 0: #No hay ninguna mascara para esta clase o use_masks=False
        indices=None
        print(" clase", clase, "get_class_items_with_mask: casos_mascara",0)
    else:
        indices=torch.argwhere(casos_mascara==1).squeeze() #tensor
        print(" clase", clase, "get_class_items_with_mask: casos_mascara",indices.shape)
    
    indices=indices.tolist() # Lista de enteros
        
    return indices

def get_class_distribution(dataset):
    '''
    dataset: CImgFruitsViewsDataSet
    Devuelve las probabilidades de cada label
    '''
    matriz_casos=pl_datamodule.matriz_etiquetas(dataset)
    print('matriz_casos.shape',matriz_casos.shape)
    distribution=torch.nansum(matriz_casos,dim=0)
    return distribution

class Balanced_BatchSampler(Sampler):
    '''
    Dado un CImgFruitsViewsDataSet multilabel
    devuelve batches donde se asegura la misma cantidad de etiquetas positivas de todas las clases
    
    Util para clases muy desbalanceadas
    '''
    def __init__(self,dataset):
        estadistica_clase=get_class_distribution(dataset)
        num_clases=len(estadistica_clase) 
        
        
        self.listas=[]
        self.lengths=[] 
        for k in range(num_clases): 
            lista= get_class_items(dataset,k)
            self.listas.append(lista)
            self.lengths.append(len(lista))
         
        self.dataset = dataset
        self.len = len(dataset)

                
   
    
    def barajarListas(self):
        for lista in self.listas:
            random.shuffle(lista)
        
        
    def __iter__(self):
        ''' Devuelve un epoch balanceado'''
        iteration = 0
        self.barajarListas()
        
        #batch=[]
        n=0
        
        while n <= self.len:
            iteration += 1
            # Coger secuencialemente un elemento de cada lista
            # Cada lista se recorre ciclicamente
            for count,lista in enumerate(self.listas):
                pos=iteration % self.lengths[count]
                n+=1
                yield lista[pos]
                # batch.append( lista[pos])
                # if len(batch)==self.batch_size:
                #     out=batch
                #     batch=[]
                #     yield out             
         
    def __len__(self) -> int:
        return self.len
    

class Balanced_BatchSamplerMultiLabel(Sampler):
    '''
    Dado un TomatoViewsDataSet multilabel
    devuelve batches donde se asegura la misma cantidad de etiquetas positivas de todas las clases
    
    Util para clases muy desbalanceadas
    '''
    def __init__(self,dataset,class_names=None,use_masks=None):
        estadistica_clase=get_class_distribution(dataset)
        num_clases=len(estadistica_clase) 
        print('Estadisticas_clases',estadistica_clase)
        print('Sampler Numclases=',num_clases)
        print('Sampler num instancias',len(dataset))
        print("Sampler use_masks",use_masks)
        self.listas=[]
        self.lengths=[] 
        lista= get_class_items(dataset,-1)
        self.listas.append(lista)
        self.lengths.append(len(lista))
        
        for k in range(num_clases): 
            lista= get_class_items(dataset,k)
            self.listas.append(lista)
            self.lengths.append(len(lista))

            lista = get_class_items_with_mask(dataset,class_names[k],use_masks) # lista con los elementos de la clase que tienen mascara
            if lista is not None: # si la clase no tiene mascara, no creará una lista
                if len(lista)>0:
                    self.listas.append(lista)
                    self.lengths.append(len(lista))
        
        print('Sampler  Numlistas=',len(self.listas)) 
        self.dataset = dataset
        #self.len =  2*len(dataset)
        factor=1 # Factor que multiplica a la longitud del dataset para asegurar que se seleccionan todas las imagenes de la lista mas larga
        print('factor len(dataset)',factor)
        self.len=int(factor*len(dataset))
        print('Sampler len=',self.len) 
        print('Sampler Lengths:',self.lengths)

   
    
    def barajarListas(self):
        for lista in self.listas:
            random.shuffle(lista)
        
        
    def __iter__(self):
        ''' Devuelve un minibatch balanceado'''
        iteration = 0
        self.barajarListas()
        
        #batch=[]
        n=0
        
        while n <= self.len:
            iteration += 1
            # Coger secuencialemente un elemento de cada lista
            # Cada lista se recorre ciclicamente
            for count,lista in enumerate(self.listas):
                if(self.lengths[count]==0):
                    continue
                pos=iteration % self.lengths[count]
                n+=1
                yield lista[pos]
                # batch.append( lista[pos])
                # if len(batch)==self.batch_size:
                #     out=batch
                #     batch=[]
                #     yield out             
         
    def __len__(self) -> int:
        return self.len


class Balanced_BatchSamplerGoodBad(Sampler):
    '''
    Dado un CImgFruitsViewsDataSet multilabel
    devuelve batches que devuelve el mismo número de etiquetas positivas y negativas en la última
    etiqueta ("Bad")
    '''
    def __init__(self,dataset):
        estadistica_clase=get_class_distribution(dataset)
        num_clases=len(estadistica_clase) 
        print('Estadisticas_clases',estadistica_clase)
        print('Sampler Numclases=',num_clases)
        print('Sampler num instancias',len(dataset))
        self.listas=[]
        self.lengths=[] 
        lista= get_class_items(dataset,-1)
        L=len(lista)
        lista_1=lista[::2]
        lista_2=lista[1::2]
        self.listas.append(lista_1)
        self.lengths.append(len(lista_1))   
        self.listas.append(lista_2)
        self.lengths.append(len(lista_2))
        
        for k in range(num_clases): 
            lista= get_class_items(dataset,k)
            self.listas.append(lista)
            self.lengths.append(len(lista))
        
        print('Sampler  Numlistas=',len(self.listas)) 
        self.dataset = dataset
        self.len =  2*len(dataset)
        print('Sampler len=',self.len) 
        print('Sampler Lengths:',self.lengths)

                
   
    
    def barajarListas(self):
        for lista in self.listas:
            random.shuffle(lista)
        
        
    def __iter__(self):
        ''' Devuelve un minibatch balanceado'''
        iteration = 0
        self.barajarListas()
        
        #batch=[]
        n=0
        
        while n <= self.len:
            iteration += 1
            # Coger secuencialemente un elemento de cada lista
            # Cada lista se recorre ciclicamente
            for count,lista in enumerate(self.listas):
                if(self.lengths[count]==0):
                    continue
                pos=iteration % self.lengths[count]
                n+=1
                yield lista[pos]
                # batch.append( lista[pos])
                # if len(batch)==self.batch_size:
                #     out=batch
                #     batch=[]
                #     yield out             
         
    def __len__(self) -> int:
        return self.len    