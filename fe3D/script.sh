#!/bin/bash
#PBS -l walltime=00:05:00

cd $PBS_O_WORKDIR
module load numpy/1.9.1/intel/14.0/mkl/11.1/python/2.7.8

python fe3D.py >> res_20.txt