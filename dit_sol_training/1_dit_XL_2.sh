#!/bin/bash
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
###SBATCH --gpus-per-node=1     # 1 GPU per node
#SBATCH --cpus-per-task=8     # 8 CPU per task
###SBATCH -c 8       ## Number of Cores
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=20G

#SBATCH -p general ## Partition
#SBATCH -q public  ## QOS

#SBATCH -C a100_80       
#SBATCH --time=10080   ## 5 minutes of compute

#SBATCH --job-name=az_dit
#SBATCH --output=slurm.%j.out  ## job /dev/stdout record (%j expands -> jobid)
#SBATCH --error=slurm.%j.err   ## job /dev/stderr record 
#SBATCH --export=NONE          ## keep environment clean
#SBATCH --mail-type=ALL        ## notify <asurite>@asu.edu for any job state change
#SBATCH -A grp_pshakari




echo "WHERE I AM FROM: $SLURM_SUBMIT_DIR"
echo "WHERE AM I NOW: $(pwd)"

echo "Loading Python 3 from Anaconda module"
module load mamba/latest

echo "Activating scientific computing Python environment: pytorch"
source activate DiT # Use conda instead of source activate

echo "Running example Python script"
torchrun --nnodes=8 --nproc_per_node=1 ../1_train.py \
--model DiT-XL/2 \
--data-path /data/yyang409/bowen/imagenet_feature/swin_base/patch4_window7_224/image_features_w_label_train.npz \
--global-batch-size 512 \
--results-dir /scratch/bowenxi/dit_result/DiT-XL_2_0423_4_a100 \
--num-workers 2 \
--log-every 1000 \
--ckpt-every 50_000



