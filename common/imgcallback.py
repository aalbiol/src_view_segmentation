# lightning related imports
import pytorch_lightning as pl
import torch
import wandb

from pytorch_lightning.callbacks import Callback

from pytorch_lightning.utilities.types import STEP_OUTPUT
from typing import Any, Dict, List, Optional, Type

import torch.nn.functional as F

from pytorch_lightning.utilities import rank_zero_info


import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg 

import matplotlib.pyplot as plt
import piltools
from pl_constrained_model import ConstrainedSegmentMIL
from torchvision import transforms
import pycimg
from PIL import Image
import os




def fig2pil(fig):
    # Create a FigureCanvasAgg renderer
    canvas = FigureCanvasAgg(fig)

    # Render the figure to a pixel buffer
    canvas.draw()
    canvas.print_png('log_figure.png')
    image=Image.open('log_figure.png')
    a = np.asarray(image)
    image2=Image.fromarray(a)
    image.close()
    os.remove('log_figure.png')
    return image2




def visualize_imgs(probs, batch,defect_types):
    
    
    dict_names={}
    for n,nombre in enumerate(defect_types):
        dict_names[nombre]=n
    
    mosaicimgs=piltools.createMosaicRGB(batch[:,:3,:,:])
    
    nrows = len(defect_types)
    ncols = 3

    fig, axes = plt.subplots(nrows = nrows, ncols= ncols,figsize=(10,10*nrows/ncols))
    axs=axes.flatten()
    
    count=0
    for k in range(len(defect_types)):
        axs[count].imshow(mosaicimgs)
        axs[count].set_title('Imagen')
        axs[count].axis('off')
        count+=1
        mosaicprobs=piltools.createMosaic(probs[:,(k,),:,:])
        axs[count].imshow(mosaicprobs,cmap='gray',clim=(0,1))
        axs[count].set_title(defect_types[k])
        axs[count].axis('off')
        count+=1
        
        probsbatch = batch.cpu()* probs[:,(k,),:,:].cpu()   
        mosaicmasked=piltools.createMosaicRGB(probsbatch[:,:3,:,:]) 
        axs[count].imshow(mosaicmasked)
        axs[count].set_title(defect_types[k] )
        axs[count].axis('off')
        count +=1
        
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()
        
    return fig




#@CALLBACK_REGISTRY
class ImageLogger(Callback):
    def __init__(self,  epoch_interval = 1, num_samples = 1, fname=None,model=None):
        super().__init__()
        self.epoch_inverval = epoch_interval
        self.num_samples = num_samples
        self.fname=fname

        
        self.model=model
        self.config=model.config
        self.num_channels_in=model.config['model']['num_channels_input']
        self.defect_types=self.config['model']['defect_types']
 
    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        batch: Any,
    ) -> None:
        """Called when the validation epoch ends."""
        
        
        if self.fname is  None:
            return
        if self.num_channels_in==3:
            vistas=pycimg.cimgread_tomate(self.fname)
        elif self.num_channels_in>3:
            vistas=pycimg.lee_vistas_RGBNIR_from_cimg(self.fname,None)  
        
        probs,batch=self.model.predice_fruto(vistas=vistas)
        probs.detach().cpu() 
        batch.detach().cpu()

            
        fig=visualize_imgs(probs,batch,self.defect_types)
        
        img=fig2pil(fig)
        trainer.logger.experiment.log({
            "imag_inference":[wandb.Image(img)]})
        plt.close(fig)

   


if __name__ == '__main__':
    
    batch = torch.rand(4,3,224,224)
    probs = torch.rand(4,2,224,224)
    
    fig = visualize_imgs(probs, batch, ['clase1', 'clase2'])
    
    img=fig2img(fig)
    plt.imshow(img)
    plt.show()