#!/bin/bash

# Define arrays of optional values
BATCH_SIZE=4
MELODY_STRATEGIES=("piano_roll" "instrument")
EPOCHS=200
LEARNING_RATE=0.001
TEACHER_FORCING_RATIOS=(0 0.5 1.0)

# Define other variables
MIDI_PATH="/dt/shabtaia/dt-sicpa/noam/deep-learning/Assignment 3/data/midi_files"
LYRICS_PATH="/dt/shabtaia/dt-sicpa/noam/deep-learning/Assignment 3/data/lyrics_train_set.csv"
TEST_PATH="/dt/shabtaia/dt-sicpa/noam/deep-learning/Assignment 3/data/lyrics_test_set.csv"
MODEL_SAVE_PATH_BASE="/dt/shabtaia/dt-sicpa/noam/deep-learning/Assignment 3/models"
WRITER_PATH_BASE="/dt/shabtaia/dt-sicpa/noam/deep-learning/Assignment 3/runs"
NUM_LAYERS=2

# Loop over combinations of melody_strategy and teacher_forcing
for MELODY_STRATEGY in "${MELODY_STRATEGIES[@]}"; do
  for TEACHER_FORCING in "${TEACHER_FORCING_RATIOS[@]}"; do
    MODEL_SAVE_PATH="${MODEL_SAVE_PATH_BASE}/melody_${MELODY_STRATEGY}_tf_${TEACHER_FORCING}"
    WRITER_PATH="${WRITER_PATH_BASE}/melody_${MELODY_STRATEGY}_tf_${TEACHER_FORCING}"
    sbatch main_job.sh \
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

  done
done
