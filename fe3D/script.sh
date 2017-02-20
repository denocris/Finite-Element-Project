#!/bin/bash
#PBS -l walltime=01:00:00

cd $PBS_O_WORKDIR
module load numpy/1.9.1/intel/14.0/mkl/11.1/python/

python fe3D.py >> res.txt