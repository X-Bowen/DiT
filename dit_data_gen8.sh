#!/bin/bash 
#SBATCH -p htc ## Partition
#SBATCH -q private  ## QOS
#SBATCH -c 16       ## Number of Cores
#SBATCH --time=240   ## 5 minutes of compute
#SBATCH --gres=gpu:h100:4
#SBATCH --job-name=data_gen
#SBATCH --output=slurm.%j.out  ## job /dev/stdout record (%j expands -> jobid)
#SBATCH --error=slurm.%j.err   ## job /dev/stderr record 
#SBATCH --export=NONE          ## keep environment clean
#SBATCH --mail-type=ALL        ## notify <asurite>@asu.edu for any job state change
#SBATCH -A grp_pshakari
#SBATCH --mem=32G



echo "WHERE I AM FROM: $SLURM_SUBMIT_DIR"
echo "WHERE AM I NOW: $(pwd)"

echo "Loading Python 3 from Anaconda module"
module load mamba/latest

echo "Activating scientific computing Python environment: pytorch"
source activate DiT # Use conda instead of source activate

echo "Running example Python script"
~/.conda/envs/DiT/bin/python 07_data_gen_npz.py 8

## torchrun --nnodes=1 --nproc_per_node=4 1_train.py --model DiT-B/4 --data-path /data/yyang409/bowen/imagenet_feature/swin_base/patch4_window7_224/image_features_w_label_train.npz --global-batch-size 512 --results-dir /scratch/bowenxi/dit_result/DiT-L_4_0327_4_a100




