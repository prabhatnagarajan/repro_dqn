#!/bin/bash
ROM=~/repro_dqn/dqn/roms/breakout.bin
STDOUT_FILE=~/breakoutresults/stdout
python ~/repro_dqn/dqn/main.py --rom $ROM &>> $STDOUT_FILE
