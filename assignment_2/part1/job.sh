#!/bin/bash
#SBATCH --job-name=rnn_lstm_experiments
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=60000M
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1
#SBATCH --mail-user=bartsch.henning@gmail.com
module purge
module load eb
module load Python/3.6.3-foss-2017b
module load Miniconda3/4.3.27
module load cuDNN/7.0.5-CUDA-9.0.176
module load NCCL/2.0.5-CUDA-9.0.176

pip3 install --user torch torchvision
export LD_LIBRARY_PATH=/hpc/eb/Debian9/cuDNN/7.1-CUDA-8.0.44-GCCcore-5.4.0/lib64:$LD_LIBRARY_PATH

source activate dl

cd /home/lgpu0211/DL/uvadlc_practicals_2019/assignment_2/part1/

#srun python3 ./train.py --experiment=True --model_type='RNN'
srun train.py --experiment=True --model_type='LSTM'

#cp -r "$TMPDIR"/output_lgpu0211 /home/lgpu0211/DL/uvadlc_practicals_2019/assignment_2/part1/output


#module load matplotlib/2.1.1-foss-2017b-Python-3.6.3

#cp -r /home/lgpu0211/DL/uvadlc_practicals_2019/assignment_2/part1 "$TMPDIR"
#mkdir "$TMPDIR"/output_lgpu0211
#cd "$TMPDIR"/part1
