#!/bin/bash
OUTPUT_FILE=/u/prabhatn/repro_dqn/breakout1/stdout.txt
ROM=/u/prabhatn/ale/thesis/dqn/roms/breakout.bin
ARGS_OUTPUT_FILE=/u/prabhatn/repro_dqn/breakout1/args.txt
EVAL_OUTPUT_FILE=/u/prabhatn/repro_dqn/breakout1/eval.txt
CHECKPOINT_DIR=/u/prabhatn/repro_dqn/breakout1/checkpoints/
EVAL_STATES_INIT_FILE=/u/prabhatn/repro_dqn/files/initstates.txt
python dqn/main.py --rom $ROM \
--args-output-file $ARGS_OUTPUT_FILE \
--eval-output-file $EVAL_OUTPUT_FILE \
--checkpoint-dir $CHECKPOINT_DIR \
--eval-init-states-file $EVAL_STATES_INIT_FILE &>> $STDOUT_FILE
