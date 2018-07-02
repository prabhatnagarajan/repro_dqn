# Deterministic Reproducibility in Deep Reinforcement Learning

This repository contains a deterministic implementation of Deep Q-learning using PyTorch and Python 2.7.  This is done in the Arcade Learning Environment.

## Getting Started

### Prerequisites

Our recommended package manager is ``` Pipenv```.

The dependencies we had were:
* Python 2.7
* ale-python-interface (0.6) ) ([Arcade Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment) - The evaluation platform)
* numpy (1.13.1)
* opencv-python (3.4.0.12)
* pip (9.0.3)
* torch (0.3.1) ([PyTorch](https://pytorch.org/))
* torchvision (0.2.0)
* matplotlib (1.5.1)


### Setup and Training a DQN

To begin, run 

```
git clone https://github.com/prabhatnagarajan/repro_dqn.git
cd repro_dqn  
sudo ./install.sh  
```

Before starting you first experiment, there are additional things to set up.

1. **Create a folder for your experiment.** For example:

```
mkdir ~/breakoutresults
```


2. **Create a checkpoint directory.** As your deep Q-network trains, it will be frequently checkpointed. Create a directory where your network checkpoints will be stored. For example:

```
mkdir ~/breakoutresults/checkpoints
```

3. **Create a checkpoint subdirectory for your rom.** Now is the time to decide which rom/game to play. Once you have decided which rom to use, create another directory *in your checkpoint directory* named after your rom. It is essential that you use the rom name from this [list](http://yavar.naddaf.name/ale/list_of_current_games.html). **Note:** The rom name is case-sensitive and should match the name from the list. Example:

```
mkdir ~/breakoutresults/checkpoints/breakout
```

4. **Set necessary constants.** Go to the repository home.

```
vim dqn/constants.py
```
This file contains the constants, or parameters for your experiment. Most of these need not be changed, however the following constants must be changed: ```CHECKPOINT_DIR``` (checkpoint directory), ```ARGS_OUTPUT_FILE``` (the constants for your experiment will be output to this file), and ```EVAL_ARGS_OUTPUT_FILE```(the constants used during offline evaluations will be output to this file). Example:
```
CHECKPOINT_DIR="/home/ubuntu/breakoutresults/checkpoints"
ARGS_OUTPUT_FILE="/home/ubuntu/breakoutresults/args.txt"
EVAL_ARGS_OUTPUT_FILE="/home/ubuntu/breakoutresults/evalargs.txt"
``` 

5. **Set variables in the training script**. 

```
vim dqn/train.sh
```

This file will start training. In this file you specify the rom, the output file, and the file to run (```train.py```)

* ```ROM``` stores the location of the rom file. E.g.
	```
	ROM=~/repro_dqn/dqn/roms/breakout.bin
	```
	Of course, this may be in a different location depending on where you cloned the repository.
* ```STDOUT_FILE``` is the file where the program output is outputted. E.g.
	```
	STDOUT_FILE=~/breakoutresults/stdout
	```
* Set the python program to the correct path:

```
python <repo home>/dqn/train.py $ROM &>> $STDOUT_FILE
```

6. **Run the Program!**
From the repository home, 
```
pipenv shell
./dqn/train.sh
```
As your agent trains, its progress is outputted to ```STDOUT_FILE```

### Running Evaluations
Once a network has been trained, we can perform the evaluations in a separate execution.

1. **Set necessary constants.** Go to the repository home.

```
vim dqn/constants.py
```
This file contains the constants, or parameters for your experiment. Most of these need not be changed, however the following constants probably need to change: ```CHECKPOINT_DIR``` (checkpoint directory), ```ARGS_OUTPUT_FILE``` (the constants for your experiment will be output to this file), and ```EVAL_ARGS_OUTPUT_FILE```(the constants used during offline evaluations will be output to this file). Example:
```
CHECKPOINT_DIR="/home/ubuntu/breakoutresults/checkpoints"
ARGS_OUTPUT_FILE="/home/ubuntu/breakoutresults/args.txt"
EVAL_ARGS_OUTPUT_FILE="/home/ubuntu/breakoutresults/evalargs.txt"
EVAL_INIT_STATES_FILE="/home/ubuntu/repro_dqn/files/initstates.txt"
``` 

The checkpoint directory contains the networks that will be evaluated.

2. **Set variables in the evaluation script**. 

```
vim dqn/eval.sh
```

This file will start training. In this file you specify the rom, the output file, and the file to run (```eval.py```)

* ```ROM``` stores the location of the rom file. E.g.
	```
	ROM=~/repro_dqn/dqn/roms/breakout.bin
	```
	Of course, this may be in a different location depending on where you cloned the repository.
* ```STDOUT_FILE``` is the file where the program output is outputted. E.g.
	```
	STDOUT_FILE=~/breakoutresults/eval
	```
* Set the python program to the correct path:

```
python <repo home>/dqn/eval.py $ROM &>> $STDOUT_FILE
```

3. **Run the Program!**
From the repository home, 
```
pipenv shell
./dqn/eval.sh
```
As your agent is evaluated, its progress is outputted to ```STDOUT_FILE```

### Verifying Identicality of Weights

From the repository home:

1. **Choose the networks to compare.** Go to the repository home.

```
vim dqn/verify_weights.py
```
Modify the list ```CHECKPOINT_FOLDERS``` to contain the list of all the networks whose weights you want to compare. For example:
```
CHECKPOINT_FOLDERS = [
"/home/ubuntu/breakoutresults/checkpoints/breakout",
"/home/ubuntu/breakoutresults/checkpoints/breakout"]
``` 
For all networks in these checkpoint folders, the program will compare the weights of the networks until it reaches the least common epoch for all the networks (in case some networks are still training). For each epoch it compares,
it will output a boolean indicating that each layer does or does not have equivalent weights across all networks for that epoch.

At the end of the program, it outputs a boolean indicating that all compared weights were/weren't equal.

2. **Run the program.** Go to the repository home.
```
python dqn/verify_weights.py
```

## Hyperparameters

The default hyperparameters for the deterministic implementation are specified in ```constants.py```. The exploration and initialization seeds used were 125, 127, 129, 131, and 133. The remainder of the hyperparameters are (these parameters are explained in the [DQN paper](https://deepmind.com/research/dqn/)):

* Network architecture: We used the architecture from the [Nips paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf).
* Timesteps - 10 million
* Minibatch size - 32
* Agent history length - 4
* Replay Memory Size - 1 million
* Target network update frequency - 10000
* Discount Factor - 0.99
* Action Repeat - 4
* Update Frequency - 2
* Learning rate - 0.0001
* Initial Exploration - 1.0
* Final Exploration - 0.1
* Final Exploration Frame - 1 million
* Replay Start Size - 50000
* No-op max - 30
* Death ends episode (during training, end episode with loss of life) - False 

## Experimental Conditions

All of our experiments were performed under the same hardware and software conditions in AWS.
* AMI - Deep Learning Base AMI (Ubuntu) Version 4.0
* Instance Type - GPU Compute, p2.xlarge
* Software - Installed using the install script. All other software comes with the AMI.

Unfortunately, it appears that AWS continuously updates its software, and there is no guarantee that the AMI will always be available. As such, it might not be able to replicate our results exactly. However, the deterministic implementation should function correctly under most AMIs available on AWS.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

I used a number of references in building this. I apologize for any references I might have excluded.

* [Original Deepmind Code](https://sites.google.com/a/deepmind.com/dqn/)
* [deep_q_rl](https://github.com/spragunr/deep_q_rl)
* [Devsisters' DQN](https://github.com/devsisters/DQN-tensorflow)
* [Discussion](https://github.com/dennybritz/reinforcement-learning/issues/30)

## Questions

If you have any questions or difficulties, feel free to open an issue or email me at prabhatn@cs.utexas.edu.
