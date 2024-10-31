#!/bin/bash
#SBATCH -c 8              # Adjust number of CPU cores based on need and availability
#SBATCH -p gpu_quad            # Use the GPU partition
#SBATCH --gres=gpu:1      # Request 1 GPU
#SBATCH --mem=32G         # Memory request, can be adjusted based on the model
#SBATCH -t 16:00:00       # Time allocation (6 hours to provide some buffer)
#SBATCH -o large_model_%j.out  # Standard output file
#SBATCH -e large_model_%j.err   # Standard error file

pwd
source nlpenv3/bin/activate
python3 -u baseline_model.py