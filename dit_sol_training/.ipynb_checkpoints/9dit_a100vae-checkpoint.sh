#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8       ## Number of Cores
#SBATCH --gres=gpu:a100:4
#SBATCH -p general ## Partition
#SBATCH -q private  ## QOS

###SBATCH -C h100       
#SBATCH --time=10080   ## 5 minutes of compute

#SBATCH --job-name=az_dit_a100_XL2_vae
#SBATCH --output=slurm.%j.out  ## job /dev/stdout record (%j expands -> jobid)
#SBATCH --error=slurm.%j.err   ## job /dev/stderr record 
#SBATCH --export=NONE          ## keep environment clean
#SBATCH --mail-type=ALL        ## notify <asurite>@asu.edu for any job state change
#SBATCH -A grp_pshakari
#SBATCH --mem=100G



echo "WHERE I AM FROM: $SLURM_SUBMIT_DIR"
echo "WHERE AM I NOW: $(pwd)"

echo "Loading Python 3 from Anaconda module"
module load mamba/latest

echo "Activating scientific computing Python environment: pytorch"
source activate DiT # Use conda instead of source activate

echo "Running example Python script"
torchrun --nnodes=1 --nproc_per_node=4 ../1vaetrain_claude.py \
--model DiT-XL/2 \
--data-path /scratch/bowenxi/dit/vae/vae_latents_with_labels.npz \
--global-batch-size 512 \
--results-dir /scratch/bowenxi/dit_result/DiT-XL_2_0517_2h100 \
--num-workers 4 \
--log-every 5000 \
--ckpt-every 100_000

### Submitted batch job 25617900

