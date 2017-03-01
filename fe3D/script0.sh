#!/bin/bash
#PBS -l walltime=10:00:00

cd $PBS_O_WORKDIR

python fe3D_matfree.py
