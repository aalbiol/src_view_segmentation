from PIL import Image
import numpy as np
import torchvision.transforms as T
import torch

def createMosaicRGB(tensor):
    '''
    Dado un tensor de numimags x 3 x h x w
    Crea un mosaico aprox cuadrado
    '''
    
    numimags = tensor.shape[0]
    w=tensor.shape[3]
    h=tensor.shape[2]
    
    assert tensor.shape[1]==3 , "El tensor debe ser numimags x 3 x h x w"
    
    numcols=int(np.ceil(np.sqrt(numimags)))
    numrows = int(np.ceil(numimags/numcols))
    
    dst = Image.new('RGB', (numcols*w, numrows*h),color=(0,0,0))
    
    counter=0
    tensor2pil = T.ToPILImage()
    
    for row in range(numrows):
        for col in range(numcols):
            if(counter >= numimags):
                break
            vista_tensor = tensor[counter]
            pil = tensor2pil(vista_tensor)
            dst.paste(pil, (col*w, row*h))
            counter += 1
            
    return dst




def createMosaic(input):
    '''
    Dado un tensor de numimags x c x h x w
    Crea un mosaico de mosaicos aprox cuadrado
    
    El mosaico grueso tiene sqrt(numimags) de lado
    
    El mosaico fino tiene sqrt(c) de lado
    '''
    
    if not isinstance(input,torch.Tensor):
        tensor = torch.tensor(input)
    else:
        tensor=input
   
    numimags = tensor.shape[0]
    w=tensor.shape[3]
    h=tensor.shape[2]
    c=tensor.shape[1]
     
    
    numcols_g=int(np.ceil(np.sqrt(numimags)))
    numrows_g = int(np.ceil(numimags/numcols_g))
    
    numcols_p=int(np.ceil(np.sqrt(c)))
    numrows_p = int(np.ceil(c/numcols_p))
    
    wg = numcols_p * w
    hg = numrows_p * h
    
    dst = Image.new('RGB', (numcols_g*numcols_p*w, numrows_g*numrows_p*h), color=(0,0,0))
    
    counter_g=0
    
    tensor2pil = T.ToPILImage()
    resize = T.Resize((w,h),)
    
    for row in range(numrows_g):
        for col in range(numcols_g):
            if(counter_g >= numimags):               
                break
            counter_p=0
            for rowp in range(numrows_p):
                for colp in range(numcols_p):  
                    if counter_p >= c:
                        break                   
                    vista_tensor = tensor[counter_g, (counter_p,),:,:] # shape 1 x h x w
                    vista_tensor = torch.tile(vista_tensor,(3,1,1))  # shape 3 x h x w  
                    vista_tensor = resize(vista_tensor)                                   
                    pil = tensor2pil(vista_tensor)
                    counter_p += 1
                    dst.paste(pil, (col*wg+colp*w, row*hg+rowp*h))
            
            counter_g += 1
            
    return dst

            
def createMosaicMono(a):
    '''
    Dado un tensor de numimags x  h x w 
    correspondiente a imagenes monocromas
    Crea un mosaico aprox cuadrado
    '''

    #tensor=np.clone(a)
    tensor=torch.clone(a)
    assert tensor.ndim==3 , "El tensor debe ser numimags x  h x w"

    numimags = tensor.shape[0]
    w=tensor.shape[2]
    h=tensor.shape[1]



    numcols=int(np.ceil(np.sqrt(numimags)))
    numrows = int(np.ceil(numimags/numcols))

    dst = np.zeros((h*numrows, w*numcols))

    counter=0


    for row in range(numrows):
        for col in range(numcols):
            if(counter >= numimags):
                break
            inirow=row*h
            endrow= inirow+h

            inicol = col*w
            endcol = inicol + w

            vista_tensor = tensor[counter]
            vista_tensor[0,:]=0
            vista_tensor[:,0]=0

            dst[inirow:endrow,inicol:endcol]=vista_tensor
            counter += 1

    return dst


def createMosaicMonoNP(a):
    '''
    Dado un tensor de numimags x  h x w 
    correspondiente a imagenes monocromas
    Crea un mosaico aprox cuadrado
    '''
    tensor=a.copy()
    assert tensor.ndim==3 , "El tensor debe ser numimags x  h x w"
    
    numimags = tensor.shape[0]
    w=tensor.shape[2]
    h=tensor.shape[1]
    
    
    
    numcols=int(np.ceil(np.sqrt(numimags)))
    numrows = int(np.ceil(numimags/numcols))
    
    dst = np.zeros((h*numrows, w*numcols))
    
    counter=0
   
    
    for row in range(numrows):
        for col in range(numcols):
            if(counter >= numimags):
                break
            inirow=row*h
            endrow= inirow+h
            
            inicol = col*w
            endcol = inicol + w
            
            vista_tensor = tensor[counter]
            vista_tensor[0,:]=0
            vista_tensor[:,0]=0
            
            dst[inirow:endrow,inicol:endcol]=vista_tensor
            counter += 1
            
    return dst
    