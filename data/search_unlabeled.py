from glob import glob
import os
import sys

import random
import yaml

import sys
current_file_dir = os.path.dirname(os.path.abspath(__file__))

# Define the relative path to the folder you want to add

common_folder = os.path.join(current_file_dir, "../common")
sys.path.append(common_folder)
import json

def analiza_anotaciones(jsonfile,defect_types):
    # load json file
    
    with open(jsonfile) as f:
        data=json.load(f)
    
    if 'annotations' not in data:
        print(jsonfile,' without annotations')
        return True
    anotaciones=data['annotations']
    if len(anotaciones) < len(defect_types):
        print(jsonfile,' without all defect types', len(anotaciones), len(defect_types))
        return True
    
    for a in defect_types:
        if a not in anotaciones:
            print(jsonfile,' without defect type:',a)
            return True
        else:
            if int(anotaciones[a])< 0:
                print(jsonfile,' with label -1 at defect type:',a)
                return True
            
    
    return False


if __name__ == '__main__':
    config_file=sys.argv[1]
    
    directorios=[]
    
    with open(config_file,'r') as f:
        config=yaml.load(f,Loader=yaml.FullLoader)
    
    
    root_folder=config['data']['root_folder']
    train=config['data']['train']
    for dp in train:
        directorios.append(os.path.join(root_folder,dp[1]))
    
    val=config['data']['val']
    for dp in val:
        directorios.append(os.path.join(root_folder,dp[1]))
    directorios=list(set(directorios)) 
    
    defect_types=config['model']['defect_types']
    
    total_ficheros=0
    incompletos=0
    for d in directorios:
        patron=os.path.join(d,"*.json")
        ficheros=glob(patron)
        total_ficheros+=len(ficheros)
        print(len(ficheros))
        for f in ficheros:
            res=analiza_anotaciones(f,defect_types)
            if res:
                incompletos+=1
    print('Total ficheros:',total_ficheros)
    print('Incompletos:',incompletos)
            
