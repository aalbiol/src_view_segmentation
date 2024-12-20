@echo off
setlocal enabledelayedexpansion

REM Ruta a GIMP
set GIMP_PATH="C:\Program Files\GIMP 2\bin\gimp-2.10.exe"

REM Ruta al archivo de texto con la lista de imágenes
set IMAGE_LIST=D:\mscanData\pngs_anotados\pngsNaranjas\bueno_malo\20240219_190526_Img_20240219_NaranjaRottEncoder\images"

REM Leer el archivo línea por línea
for /f "usebackq delims=" %%i in (%IMAGE_LIST%) do (
    echo Abriendo %%i
    %GIMP_PATH% "%%i"
    timeout /t 2 > nul
)