#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16       ## Number of Cores
#SBATCH --gres=gpu:h100:4
#SBATCH -p general ## Partition
#SBATCH -q private  ## QOS

###SBATCH -C h100       
#SBATCH --time=10080   ## 5 minutes of compute

#SBATCH --job-name=az_dit_h100_L2
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
--model DiT-L/4 \
--data-path /data/yyang409/bowen/imagenet_feature/swin_base/patch4_window7_224/image_features_w_label_train.npz \
--global-batch-size 256 \
--results-dir /scratch/bowenxi/dit_result/DiT-L_4_0518_4h100 \
--num-workers 12 \
--log-every 5000 \
--ckpt-every 100_000 \
--resume /scratch/bowenxi/dit_result/DiT-L_2_0513_4h100/000-DiT-L-2/checkpoints/0550000.pt

### Submitted batch job 25617900

