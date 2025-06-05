#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16       ## Number of Cores
#SBATCH --gres=gpu:h100:4
#SBATCH -p general ## Partition
#SBATCH -q private  ## QOS

###SBATCH -C a100_80       
#SBATCH --time=10080   ## 5 minutes of compute

#SBATCH --job-name=az_dit_xl_4
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
torchrun --nnodes=1 --nproc_per_node=4 ../1_train_resume.py \
--model DiT-XL/4 \
--data-path /data/yyang409/bowen/imagenet_feature/swin_base/patch4_window7_224/image_features_w_label_train.npz \
--global-batch-size 256 \
--results-dir /scratch/bowenxi/dit_result/DiT-XL_4_0530_4h100 \
--num-workers 12 \
--log-every 10000 \
--ckpt-every 100_000 \
--resume /scratch/bowenxi/dit_result/DiT-XL_4_0522_4h100/000-DiT-XL-4/checkpoints/4700000.pt

### Submitted batch job 25617900

