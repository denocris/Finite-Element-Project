#!/bin/bash
#PBS -l walltime=01:00:00

cd $PBS_O_WORKDIR

python fe3D.py >> res.txt