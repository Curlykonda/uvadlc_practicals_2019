#!/bin/bash
#SBATCH --job-name=lstm_generator_alice
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
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
module load matplotlib/2.1.1-foss-2017b-Python-3.6.3
pip3 install --user torch torchvision
export LD_LIBRARY_PATH=/hpc/eb/Debian9/cuDNN/7.1-CUDA-8.0.44-GCCcore-5.4.0/lib64:$LD_LIBRARY_PATH
export PYTHONIOENCODING=utf8

source activate dl
#mkdir "$TMPDIR"/work_lgpu0211
#cp -r /home/lgpu0211/DL/uvadlc_practicals_2019/assignment_2/part2 "$TMPDIR"/work_lgpu0211

#cd "$TMPDIR"/work_lgpu0211/part2/
cd /home/lgpu0211/DL/uvadlc_practicals_2019/assignment_2/part2/
ls

python3 train.py --txt_file='./assets/book_EN_grimms_fairy_tails.txt' --book_name='grimm' --train_steps=100000 --sample_every=1000 --snippet_completion=True --snippet='Sleeping beauty is'

