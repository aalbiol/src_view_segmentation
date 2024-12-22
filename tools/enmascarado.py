import subprocess
import time
import os

# Ruta a GIMP
gimp_path = r"C:\Program Files\GIMP 2\bin\gimp-2.10.exe"

# Ruta al archivo de texto con la lista de imágenes
image_list_path = r"D:\mscanData\pngs_anotados\pngsNaranjas\bueno_malo\20240312_210950_prod\20240805\images\cped_masks_malos.txt"

# Leer el archivo línea por línea
with open(image_list_path, 'r') as file:
    images = file.readlines()

# Abrir cada imagen con GIMP
for image in images:
    image = image.strip()  # Eliminar espacios en blanco y saltos de línea
    if image:  # Asegurarse de que la línea no esté vacía
        if os.path.exists(image.replace('_RGB.png','_RGB_mask.png')):
            print(f"Abriendo {image.replace('_RGB.png','_RGB_mask.png')}")
            subprocess.Popen([gimp_path, image.replace('_RGB.png','_RGB_mask.png')])
        print(f"Abriendo {image}")
        subprocess.Popen([gimp_path, image],shell=True)
        # Espera a que el usuario presione una tecla
        input('Presione Enter para continuar...')
        
       

    else:
        print(f'La imagen {imagen} no se encontró.')