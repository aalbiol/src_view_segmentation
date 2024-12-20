import torch
from torchvision import transforms


class Aumentador_Imagenes_y_Mascaras():
    def __init__(self, geometric_transforms, color_transforms,normalize_transforms):
        self.geometric_transforms = geometric_transforms
        self.color_transforms = color_transforms
        self.normalize_transforms=normalize_transforms

    def __call__(self, img, masks):
        ncanales=img.shape[0]
        #if ncanales is not 3:
         #   img1 = self.color_transforms(img[0:3]) # Solo a la imagen
          #  img1=torch.cat(img1,img[-1])
        #else:
        if self.color_transforms is not None:
            RGB=img[:3,:,:]
            RGB1 = self.color_transforms(RGB) # Solo a la imagen
            if img.shape[0] == 4 :
                canales_extra=img[3:,:,:]
                canales_extra1=self.color_transforms(canales_extra)
                img1=torch.concat([RGB1,canales_extra1],dim=0)
            elif img.shape[0] == 5 :
                canal_extra1=img[3:4,:,:]
                canal_extra1_transformed=self.color_transforms(canal_extra1)
                canal_extra2=img[4:5,:,:]
                canal_extra2_transformed=self.color_transforms(canal_extra2)           
                img1=torch.concat([RGB1,canal_extra1_transformed],dim=0)
                img1=torch.concat([img1,canal_extra2_transformed],dim=0)
            elif img.shape[0]==6:
                canal_extra=img[3:6,:,:]
                canal_extra_transformed=self.color_transforms(canal_extra)
                img1=torch.concat([RGB1,canal_extra_transformed],dim=0)
            else:
                img1=RGB1
        else:
            img1=img
        img1=self.normalize_transforms(img1)
        
        taco=img1
        for m in masks: # Creamos un taco con la imagen y las máscaras
            if m is not None:
                taco=torch.cat((taco,m.unsqueeze(0)),0)
        taco=self.geometric_transforms(taco)
        
        img2=taco[:ncanales,:,:]       
        n=0
        masks2=[]
        for k in range(len(masks)):
            m=masks[k]
            if m is not None:
                masks2.append(taco[ncanales+n,:,:])
                n+=1
            else:
                masks2.append(None)
            
        return img2, masks2