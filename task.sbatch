#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J crack_detection
#SBATCH -o ./OUTPUT/crack_detection.%J.out
#SBATCH -e ./OUTPUT/crack_detection.%J.err
#SBATCH --mail-user=yongxiang.shi@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=02:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
# need time, and resources will not be released after program stops.

module load tensorflow/1.13.1-cuda10.0-cudnn7.6-py3.6
module load keras/2.2.4-cuda10.0-cudnn7.6-py3.6
module load anaconda3/4.4.0

#run the application:
python trainCNN.py
#python do_test.py
