import warnings

from pathlib import Path
from argparse import ArgumentParser
import argparse
import os
import sys
warnings.filterwarnings('ignore')

# get path to current file directory

current_file = Path(os.path.abspath(__file__))

current_dir = os.path.dirname(current_file)


sys.path.append(os.path.join(current_dir,'../common'))

# torch and lightning imports
import torch
torch.set_float32_matmul_precision('medium')
torch.multiprocessing.set_sharing_strategy('file_system')


import matplotlib.pyplot as plt
import numpy as np

import yaml

from pl_constrained_model import ConstrainedSegmentMIL

from glob import glob


import matplotlib.pyplot as plt
from tqdm import tqdm

# Backend pdf matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
#matplotlib.use('pdf')

import collections
import json
#matplotlib.use('TkAgg')

def split_list(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]

###################################################
# Genera un pdf con resultados de segmentacion por cada carpeta #
# Recibe como argumentos una lista de carpetas #
###################################################


if __name__ == '__main__':

    if len(sys.argv)<3:
        print('Uso: python predict  config  fich1.npz fich2.npz ...')
        sys.exit(1)

    configfile=sys.argv[1]
    with open(configfile,'r') as f:
        config=yaml.load(f,Loader=yaml.FullLoader)    

    configpredict=config['predict']
    modelofilename = os.path.join(configpredict['model_dir'],configpredict['model_file'])
    
    npzs=sys.argv[2:]


  
    defectos=configpredict['defectos']
    if len(defectos)!=3:
        print('Error: debe haber 3 defectos')
        print('defectos:',defectos) 
        sys.exit(1)

    print('Modelo:',modelofilename)

    model = ConstrainedSegmentMIL()
    model.load(modelofilename)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    reportsdir="predictions"
    if not os.path.exists(reportsdir):
        print("Creating reports directory: ",reportsdir)
        Path( reportsdir ).mkdir( parents=True, exist_ok=True )

    for npz in npzs:
        resultados=model.predictnpzs(npz,device,include_images=True)
        print('Resultados:',len(resultados))
        for res in resultados:
            rgb=res['img'][:,:,:3]
            dictprobs=res['pixel_probs_dict']
            probss=[]
            for defecto in defectos:
                probss.append(dictprobs[defecto])
            probs=np.stack(probss,axis=2)
            
            _=plt.subplot(121)
            plt.imshow(rgb)
            plt.title('RGB')
            _=plt.subplot(122)
            plt.imshow(probs,cmap='gray',clim=[0,1])
            plt.title('Probabilidad ' + "+".join(defectos))
            plt.show()
            #plt.close()
            print("rgbshape",rgb.shape)
            print("probs shape",probs.shape)
            print (res.keys())

        

            
            
            