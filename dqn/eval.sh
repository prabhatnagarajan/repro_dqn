#!/bin/bash
ROM=~/repro_dqn/dqn/roms/breakout.bin
STDOUT_FILE=~/breakoutresults/eval
python ~/repro_dqn/dqn/eval.py $ROM &>> $STDOUT_FILE
