#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --partition main			### specify partition name where to run a job. main: all nodes; gtx1080: 1080 gpu card nodes; rtx2080: 2080 nodes; teslap100: p100 nodes; titanrtx: titan nodes
#SBATCH --time 1-10:30:00			### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name job_1			### name of the job
#SBATCH --output '/dt/shabtaia/dt-sicpa/noam/deep-learning/Assignment 3/out_jobs/%x-%J.out'			### output log for running job - %J for job number
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


python_script='/dt/shabtaia/dt-sicpa/noam/deep-learning/Assignment 3/src/main.py'


while [[ "$#" -gt 0 ]]; do
    case $1 in
        --midi_path) MIDI_PATH="$2"; shift ;;
        --lyrics_path) LYRICS_PATH="$2"; shift ;;
        --test_path) TEST_PATH="$2"; shift ;;
        --model_save_path) MODEL_SAVE_PATH="$2"; shift ;;
        --writer_path) WRITER_PATH="$2"; shift ;;
        --num_layers) NUM_LAYERS="$2"; shift ;;
        --learning_rate) LEARNING_RATE="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --epochs) EPOCHS="$2"; shift ;;
        --melody_strategy) MELODY_STRATEGY="$2"; shift ;;
        --teacher_forcing) TEACHER_FORCING="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo

# Run the Python script with the parsed arguments

python "$python_script" \
  --midi_path "$MIDI_PATH" \
  --lyrics_path "$LYRICS_PATH" \
  --test_path "$TEST_PATH" \
  --model_save_path "$MODEL_SAVE_PATH" \
  --writer_path "$WRITER_PATH" \
  --num_layers $NUM_LAYERS \
  --learning_rate $LEARNING_RATE \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --melody_strategy $MELODY_STRATEGY \
  --teacher_forcing $TEACHER_FORCING
