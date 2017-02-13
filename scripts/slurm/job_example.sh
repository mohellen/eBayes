#!/bin/bash

#SBATCH -o /home/hpc/pr63so/di29zaf/workspace/statinva/batch_slurm/statinva.%j.out
#SBATCH -D /home/hpc/pr63so/di29zaf/workspace/statinva/batch_slurm
#SBATCH -J statinvasg
#SBATCH --get-user-env
#SBATCH --partition=snb
#SBATCH --exclude=mac-nvd01,mac-nvd02,mac-nvd03,mac-nvd04,mac-ati01,mac-ati02,mac-ati03,mac-ati04
#SBATCH --nodes=8
#SBATCH --tasks-per-node=32
#SBATCH --mail-type=end
#SBATCH --mail-user=hellenbr@in.tum.de
#SBATCH --export=NONE
#SBATCH --time=23:00:00


source /etc/profile.d/modules.sh
export LD_LIBRARY_PATH=/home/hpc/pr63so/di29zaf/workspace/statinva/lib/SGpp/lib/sgpp:$LD_LIBRARY_PATH 

cd ../
module load scons
module load python
module load gcc/4.9
make sgpp

mpiexec -n 256 ./bin/run > output/runtime.info
