# INF-351-LAB1


## Incluye
---
* Todos los archivos primarios (.cu)
* Este README
* Makefile
* imagen.txt

## Dependencias
---
* iostream
* time.h
* cuda_runtime.h
* stdio.h

## Ejecucion
---
Para compilar, se debe utilizar el comando de make:

$ make

Luego, para ejecutar se debe utilizar:

$ time ./output

en caso de usar un archivo de entrada, utilizar:

$ time ./output < imagen.txt

Se recomienda eliminar cualquier dependecia antes y despues de la ejecucion:

$ make clean

## Datos de Entorno
---
* SO: Ubuntu 16.04.6 LTS 
* Kernel : 4.4.0-1079-aws
* nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2017 NVIDIA Corporation
Built on Fri_Sep__1_21:08:03_CDT_2017
Cuda compilation tools, release 9.0, V9.0.176
* GNU Make 4.1


## Detalles de la instancia de AWS
---
* Number of CUDA devices 1.
* There is 1 device supporting CUDA
* For device #0
* Device name:                GRID K520
* Major revision number:      3
* Minor revision Number:      0
* Total Global Memory:        -61603840
* Total shared mem per block: 49152
* Total const mem size:       65536
* Warp size:                  32
* Maximum block dimensions:   1024 x 1024 x 64
* Maximum grid dimensions:    2147483647 x 65535 x 65535
* Clock Rate:                 797000
* Number of muliprocessors:   8

## Datos Programadores
---
Nombre:	Monserrat Figueroa Lagos
ROL: 201573525-5
Correo:	monserrat.figueroa@sansano.usm.cl

Nombre:	Camilo Nunez Fernandez
ROL: 	201573573-5
Correo:	camilo.nunezf@sansano.usm.cl
