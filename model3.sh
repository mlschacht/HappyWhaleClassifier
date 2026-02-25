#!/bin/bash

#SBATCH --account=bgmp                    #REQUIRED: which account to use
#SBATCH --partition=bgmp                  #REQUIRED: which partition to use
#SBATCH --cpus-per-task=10                 #optional: number of cpus, default is 1
#SBATCH --job-name=model3           #optional: job name
#SBATCH --output=./outfiles/model3_%j.out       #optional: file to store stdout from job, %j adds the assigned jobID
#SBATCH --error=./errorfiles/model3_%j.err        #optional: file to store stderr from job, %j adds the assigned jobID
#SBATCH --time=0-10:00:00                  ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --mem=60G                # Total memory for the job

conda activate /projects/bgmp/shared/Bi625/ML_Assignment/Conda_Envs/HumpbackClassifierEnv

/usr/bin/time -v ./model3.py