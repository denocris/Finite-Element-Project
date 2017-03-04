#!/bin/bash
#PBS -l nodes=1:ppn=20
#PBS -l walltime=10:00:00

cd $PBS_O_WORKDIR

python fe3D_p12.py
