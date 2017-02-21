#!/bin/bash
#PBS -l walltime=10:00:00

cd $PBS_O_WORKDIR

python fe3D.py >> res_20.txt
