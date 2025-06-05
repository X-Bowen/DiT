#!/bin/bash 
#SBATCH -p general ## Partition
#SBATCH -q public  ## QOS
#SBATCH -c 32       ## Number of Cores
#SBATCH --time=10080   ## 5 minutes of compute
#SBATCH --gres=gpu:a100:4
#SBATCH --job-name=az_dit
#SBATCH --output=slurm.%j.out  ## job /dev/stdout record (%j expands -> jobid)
#SBATCH --error=slurm.%j.err   ## job /dev/stderr record 
#SBATCH --export=NONE          ## keep environment clean
#SBATCH --mail-type=ALL        ## notify <asurite>@asu.edu for any job state change
#SBATCH -A grp_yyang409
#SBATCH --mem=100G



echo "WHERE I AM FROM: $SLURM_SUBMIT_DIR"
echo "WHERE AM I NOW: $(pwd)"

echo "Loading Python 3 from Anaconda module"
module load mamba/latest

echo "Activating scientific computing Python environment: pytorch"
source activate DiT # Use conda instead of source activate

echo "Running example Python script"
torchrun --nnodes=1 --nproc_per_node=4 1_train.py --model DiT-B/4 --data-path /data/yyang409/bowen/imagenet_feature/swin_base/patch4_window7_224/image_features_w_label_train.npz --global-batch-size 512 --results-dir /scratch/bowenxi/dit_result/DiT-L_4_0327_4_a100




