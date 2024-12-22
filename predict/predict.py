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
matplotlib.use('pdf')

import collections
import json


def split_list(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]

###################################################
# Genera un pdf con resultados de segmentacion por cada carpeta #
# Recibe como argumentos una lista de carpetas #
###################################################


if __name__ == '__main__':

    if len(sys.argv)<3:
        print('Uso: python predict config_file  directorio1 directorio2 ...')
        sys.exit(1)

    configfile=sys.argv[1]
    with open(configfile,'r') as f:
        config=yaml.load(f,Loader=yaml.FullLoader)    

    configpredict=config['predict']
    modelofilename = os.path.join(configpredict['model_dir'],configpredict['model_file'])
    patron_archivos=configpredict['patron']
    directorios=sys.argv[2:]


  

    print('Modelo:',modelofilename)

    model = ConstrainedSegmentMIL()
    model.load(modelofilename)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    reportsdir=configpredict['report_dir']
    if not os.path.exists(reportsdir):
        print("Creating reports directory: ",reportsdir)
        Path( reportsdir ).mkdir( parents=True, exist_ok=True )

    for directorio in directorios:
        print('\n >>> Directorio a evaluar:',directorio)
        directotorio_plano=directorio.replace('/','_')
        directotorio_plano=directotorio_plano.replace("\\",'_')
        directotorio_plano=directotorio_plano.replace('..','')
        print('directorio_plano',directotorio_plano)
        patron=os.path.join(directorio,patron_archivos)
        print('Patron:',patron)
        imagenes=glob(patron)
        imagenes.sort()
        resultados_directorio=[]
        listas=split_list(imagenes,10)
        for sublista in tqdm(listas):
            #print("sublista:",sublista)
            resultados=model.predict(sublista,device,include_images=False)
            for res in resultados:
                resultados_directorio.append(res)
        res_dict={}
        print("Numero de imagenes:",len(resultados_directorio))
        for res in resultados_directorio:
            res_dict[res['imgname']]=res['view_probs']
        # Guardar res_dict en un archivo json
        res_dict = collections.OrderedDict(sorted(res_dict.items()))
        #print(res_dict)
        jsonname=os.path.join(reportsdir,os.path.basename(directotorio_plano)+'.json')
        print("Writing json file:",jsonname)
        with open(jsonname,'w') as f:
            json.dump(res_dict,f,indent=3)
        
    # for directorio in directorios:
    #     print('Directorio:',directorio)
    #     imagenes=glob(directorio+'/*.jpg')
    #     imagenes.sort()
    #     resultados=model.predict(imagenes,device=device)
    #     directoriobasename=os.path.basename(directorio)
    #     pdfname=os.path.join(reportsdir,directoriobasename+'.pdf')
    #     areas=[]
    #     print('Generando reporte:',pdfname)
    #     with PdfPages(pdfname) as pdf:
    #         for i in tqdm(range(len(imagenes))):
    #             marcada=resultados[i]['marcada']
    #             nombre=resultados[i]['imgname']
    #             nombre=os.path.basename(nombre)
    #             area_remolque=np.sum(resultados[i]['probs_pixeles'][1])
    #             area_lodo=np.sum(resultados[i]['probs_pixeles'][0])
    #             area_imagen=marcada.shape[0]*marcada.shape[1]
    #             fraccion_remolque=area_remolque/area_imagen * 100.0
    #             fraccion_lodo=area_lodo/(area_remolque+area_lodo)*100.0
    #             areas.append([area_remolque,area_lodo,area_imagen])
    #             leyenda=' Semantic: remolque: {:.1f} %, lodo: {:.1f} %'.format(fraccion_remolque,fraccion_lodo)
               
    #             plt.imshow(marcada)
    #             plt.title(leyenda)
    #             plt.xlabel(nombre)
    #             pdf.savefig()
    #             plt.close()
    #         areas=np.array(areas)
    #         npyname=pdfname.replace('.pdf','.npy')
    #         np.save(npyname,areas)
    #         plt.plot(areas)
    #         plt.legend(['remolque','lodo','imagen'])
    #         plt.title('Areas ' + directoriobasename)
    #         pdf.savefig()
    #         plt.close()
    #         fraccion_lodo=areas[:,1]/(areas[:,0]+areas[:,1])*100.0
    #         fraccion_remolque=areas[:,0]/areas[:,2]*100.0
    #         plt.plot(fraccion_lodo,label='fracc lodo')
    #         plt.plot(fraccion_remolque,label='fracc remolque')
    #         plt.title('Fracciones ' + directoriobasename)
    #         pdf.savefig()
    #         plt.close()
            
            
            