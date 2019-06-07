#!/bin/bash
#SBATCH -J hres_Lc
#SBATCH --mem=20000
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1

module load anaconda/5.2.0/python-3.6 intel/17.0.3
export OMP_NUM_THREADS=1

export MY_WORK_DIR=$PWD
cp -pr * $SLURM_TMPDIR
cd $SLURM_TMPDIR
bash -c '(while true; do sleep 1h; rsync -a --exclude '*.tmp' $SLURM_TMPDIR/ $MY_WORK_DIR/ ; echo "Synchronized at: $(date)"; done)' &
srun python ./launch_dynamics.py
rsync -a --exclude '*.tmp' * $MY_WORK_DIR/
