#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --partition main			### specify partition name where to run a job. main: all nodes; gtx1080: 1080 gpu card nodes; rtx2080: 2080 nodes; teslap100: p100 nodes; titanrtx: titan nodes
#SBATCH --time 1-10:30:00			### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name n_collect_ffhq			### name of the job
#SBATCH --output '/dt/shabtaia/dt-sicpa/noam/deep-learning/Assignment 2/jobs_output/%x-%J.out'			### output log for running job - %J for job number
#SBATCH --gpus=rtx_6000:1				### number of GPUs, allocating more than 1 requires IT team's permission

# Note: the following 4 lines are commented out
##SBATCH --mail-user=user@post.bgu.ac.il	### user's email for sending job status messages
##SBATCH --mail-type=ALL			### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
##SBATCH --mem=48G				### ammount of RAM memory, allocating more than 60G requires IT team's permission

################  Following lines will be executed by a compute node    #######################

### Print some data to output file ###
echo date
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"


### Start your code below ####
module load anaconda				### load anaconda module (must be present when working with conda environments)
source activate /dt/shabtaia/dt-sicpa/envs/diffusion_envs


python_script='/dt/shabtaia/dt-sicpa/noam/deep-learning/Assignment 2/main.py'

python "$python_script"