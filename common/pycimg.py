#
#  File            : pycimg.py
#                    ( Python file )
#
#  Description     : Show how to import .cimg and .cimgz files into python (numpy).
#                    This file is a part of the CImg Library project.
#                    ( http://cimg.eu )
#
#  Copyright       : Antonio Albiol, Universidad Politecnica Valencia (SPAIN)
#
#                    In case of issues or comments contact Antonio Albiol at:
#                    aalbiol (at) dcom.upv.es
#
#  Licenses        : This file is 'dual-licensed', you have to choose one
#                    of the two licenses below to apply.
#
#                    CeCILL-C
#                    The CeCILL-C license is close to the GNU LGPL.
#                    ( http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html )
#
#                or  CeCILL v2.1
#                    The CeCILL license is compatible with the GNU GPL.
#                    ( http://www.cecill.info/licences/Licence_CeCILL_V2.1-en.html )
#
#  This software is governed either by the CeCILL or the CeCILL-C license
#  under French law and abiding by the rules of distribution of free software.
#  You can  use, modify and or redistribute the software under the terms of
#  the CeCILL or CeCILL-C licenses as circulated by CEA, CNRS and INRIA
#  at the following URL: "http://www.cecill.info".
#
#  As a counterpart to the access to the source code and  rights to copy,
#  modify and redistribute granted by the license, users are provided only
#  with a limited warranty  and the software's author,  the holder of the
#  economic rights,  and the successive licensors  have only  limited
#  liability.
#
#  In this respect, the user's attention is drawn to the risks associated
#  with loading,  using,  modifying and/or developing or reproducing the
#  software by the user in light of its specific status of free software,
#  that may mean  that it is complicated to manipulate,  and  that  also
#  therefore means  that it is reserved for developers  and  experienced
#  professionals having in-depth computer knowledge. Users are therefore
#  encouraged to load and test the software's suitability as regards their
#  requirements in conditions enabling the security of their systems and/or
#  data to be ensured and,  more generally, to use and operate it in the
#  same conditions as regards security.
#
#  The fact that you are presently reading this means that you have had
#  knowledge of the CeCILL and CeCILL-C licenses and that you accept its terms.
#

import numpy as np
import zlib
import os
from PIL import Image
import torch

typesDict={'float':'float32' ,'double':'float64',
'unsigned_short':'uint16','unsigned_char':'uint8',
'int':'int32', 'short':'int16'}

def __cimgread( filename ):
    """ USAGE: a= cimgread(filename)
    For CImg Images:
        * returns a npy array in the case of cimg
        * Supports compression
        * It squeezes singleton dimensions. If a CImg image has dimensions (w,h,1,c)
            the returned python object will have shape
                a.shape --> (h,w,c)
        * a(y,x,z,c) to access one element
    For CImgList:
        * returns a list of npy arrays
        * if original CImgList has nimages, then
             len(a) --> nimages
        * To access one pixel of the j-th image use a[j](y,x,z,c)

        """

    basename, file_extension = os.path.splitext(filename)
    fa = open(filename, 'rb')

    out =[]
    line0 = fa.readline() #Endiannes
    tiposdato=line0.split()
    number_of_images=int(tiposdato[0])
    datatypecimg=tiposdato[1].decode()
    endiannes = tiposdato[2]

    datatype = typesDict[datatypecimg];

    for n in range(number_of_images):
        line1 = fa.readline() # Dimensions
        dimensiones = line1.split()
        width = int(dimensiones[0]);
        height = int(dimensiones[1]);
        depth = int(dimensiones[2]);
        spectrum = int(dimensiones[3]);    

        if file_extension == '.cimgz':
            csize= int(dimensiones[4].decode()[1:])
            data = fa.read(csize)
            data = zlib.decompress(data)
        else:
            data = fa.read(width*height*depth*spectrum*np.dtype(datatype).itemsize)

        flattened = np.frombuffer(data,dtype=datatype)

        cimg=flattened.reshape((spectrum,depth,height,width))
        cimg=np.squeeze(np.transpose(cimg,(2,3,1,0)))
        out.append(cimg)

    fa.close()
    if len(out)==1:
        return out[0]
    return out

def cimgread_tomate(filename):
    a=__cimgread(filename)
    apil=[]

    
    for aa in a:
        pil=Image.fromarray(aa[:,:,:4].astype(np.uint8))
        apil.append(pil)
    
    return apil

def lee_vistas_RGBNIR_from_cimg(filename,crop_size):
    a=__cimgread(filename)
    apil=[]
    for aa in a:
        canales=[]
        
        if aa.shape[2]==4:
            rgb=aa[:,:,:3].astype('float32')/255
            rgb[rgb>1]=1
            rgb=torch.tensor(rgb)
            canales.append(rgb.permute(2,0,1))
            nir=aa[:,:,3:].astype('float32')/1023
            nir[nir>1]=1
            nir=torch.tensor(nir)
            canales.append(nir.permute(2,0,1))
        elif aa.shape[2]==6:
            rgb=aa[:,:,:3].astype('float32')/255
            rgb[rgb>1]=1
            rgb=torch.tensor(rgb)
            canales.append(rgb.permute(2,0,1))
            nir=aa[:,:,3:4].astype('float32')/255
            nir[nir>1]=1
            nir=torch.tensor(nir)
            canales.append(nir.permute(2,0,1))
        elif aa.shape[2]==5:
            rgb=aa[:,:,:3].astype('float32')/255
            rgb[rgb>1]=1
            rgb=torch.tensor(rgb)
            canales.append(rgb.permute(2,0,1))
            nir=aa[:,:,3:4].astype('float32')/1023
            nir[nir>1]=1
            nir=torch.tensor(nir)
            canales.append(nir.permute(2,0,1))
            uv=aa[:,:,4:5].astype('float32')/1023
            uv[uv>1]=1
            uv=torch.tensor(uv)
            canales.append(uv.permute(2,0,1))
        elif aa.shape[2]==3:
            print('No hay NIR, solo RGB')
            im=aa[:,:,:].astype('float32')/255
            im[im>1]=1
            canales.append(torch.tensor(im).permute(2,0,1))
        canales=torch.concat(canales,0)
        if crop_size is not None:
            h=canales.shape[1]
            w=canales.shape[2]
            fila_ini= int((h - crop_size[0])//2)
            fila_fin=-fila_ini
            col_ini=int(w-crop_size[1])//2
            col_fin=-col_ini
            canales=canales[:,fila_ini:fila_fin,col_ini:col_fin]
        apil.append(canales)     
    return apil

def cimgread(filename):
    a=__cimgread(filename)
    apil=[]

    
    for aa in a:
        pil=Image.fromarray(aa[:,:,:3])
        apil.append(pil)
    
    return apil

def cimgread_np(filename):
    a=__cimgread(filename)  
    return a    