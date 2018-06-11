import numpy as np
import torch

##################################################
##            Determinism parameters            ##
##################################################
'''
An ALE stochasticity parameter that determines the
probability that agent repeats its prior action. 
Should be set to 0 for a deterministic implementation.
'''

REPEAT_ACTION_PROBABILITY=0.0
REPEAT_ACTION_SEED = 123

'''
Random seed for the Arcade Learning Environment
'''
ALE_SEED=123
'''
A flag to turn on Determinism in the GPU
'''
GPU_DETERMINISM=True
'''
A random seed to seed a random number generator that 
controls exploration
'''
EXPLORATION_SEED=123
'''
A random seed to seed a random number generator for 
exploration to control no-ops. A random number of 
No-ops are performed at the beginning of each episode
in order to randomize the initial start state.
'''
NO_OP_SEED=123
'''
As with most deep learning algorithms, Deep Q-learning
use random minibatch sampling. This seed is for a random
number generator that controls the indices of the samples.
'''
REPLAY_SAMPLE_SEED=123
'''
Create random number generators for each of exploration,
random no-ops, and minibatch sampling, all sources of 
randomness within deep Q-learning.
'''
RNDEXP=np.random.RandomState(EXPLORATION_SEED)
RNDNO_OP=np.random.RandomState(NO_OP_SEED)
RND_MEMSAMPLE=np.random.RandomState(REPLAY_SAMPLE_SEED)
RND_ACTREPEAT=np.random.RandomState(REPEAT_ACTION_SEED) 

'''
A network seed for PyTorch. This allows us to control
the weight initialization of the neural network.
'''
NETWORK_SEED=123
#Pytorch
torch.manual_seed(NETWORK_SEED)
if torch.cuda.is_available():
    print "Using cuda....."
    torch.cuda.manual_seed_all(NETWORK_SEED)
    torch.backends.cudnn.deterministic = GPU_DETERMINISM
else:
    print "Not using cuda..."

##################################################
##             Experiment parameters            ##
##################################################
'''
DQN was originally introduced in 2013, with a Nature
paper in 2015. The two papers used different architectures.
The NATURE parameter determines which architecture to use.
Source: 
Nature paper: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
Nips paper: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
'''
NATURE=False
USE_NO_OPS=True
##################################################
##             Algorithm parameters             ##
##################################################
TRAINING_FRAMES = 50000000 if NATURE else 10000000
MINIBATCH_SIZE=32
REPLAY_CAPACITY=1000000
HIST_LEN=4
TGT_UPDATE_FREQ=10000
DISCOUNT=0.99
ACT_REPEAT=4
# Number of timesteps between gradient updates.
UPDATE_FREQ = 4 if NATURE else 2
LEARNING_RATE=0.00025 if NATURE else 1e-4
#initial exploration epsilon
INITIAL_EPSILON=1.0
# the lowest epsilon used in the training process
FINAL_EPSILON = 0.01 if REPEAT_ACTION_PROBABILITY > 0.0 else 0.1
'''
In the first FINAL_EXPLORATION_FRAME frames, we anneal
the exploration epsilon from INITIAL_EPSILON to
FINAL_EPSILON
'''
FINAL_EXPLORATION_FRAME=1000000
REPLAY_START_SIZE=50000
'''
The maximum number of no-ops at
the beginning of an episode.
'''
NO_OP_MAX=30
ALPHA=0.95
MIN_SQUARED_GRADIENT=0.01 if NATURE else 1e-6
'''
The exploration epsilon used at 
test time.
'''
TEST_EPSILON = 0.005 if REPEAT_ACTION_PROBABILITY > 0.0 else 0.05
'''
In some Atari games the agent has many lives.
DEATH_ENDS_EPISODE specifies whether we should terminate
an episode on the loss of a life during training.
'''
DEATH_ENDS_EPISODE=True if NATURE else False
##################################################
##                   Files                      ##
##################################################
'''
CHECKPOINT_DIR - directory to store the network checkpoints for reloading
ARGS_OUTPUT_FILE - file to output the arguments from this file!
'''
CHECKPOINT_DIR="/home/ubuntu/breakoutresults/checkpoints"
ARGS_OUTPUT_FILE="/home/ubuntu/breakoutresults/args.txt"
EVAL_ARGS_OUTPUT_FILE="/home/ubuntu/breakoutresults/evalargs.txt"
##################################################
##            Evaluation Parameters             ##
##################################################
EVALUATION_PHASE = False # whether or not to have an evaluation phase
EVAL_FREQ = 250000 if NATURE else 100000
EVAL_STEPS = 125000
CAP_EVAL_EPISODES = True
EVAL_MAX_FRAMES = 18000
EVAL_EPISODES=100
EVAL_BATCH_SEED = 123
EVAL_EXP_SEED = 123
EVAL_NO_OP_SEED = 123
EVAL_Q_SEED  = 123
EVAL_Q_NO_OP_SEED = 123
EVAL_INIT_STATES_FILE = "/home/ubuntu/repro_dqn/files/initstates.txt"


##################################################
##            Weight storage Parameters         ##
##################################################
CHECKPOINT_FREQUENCY = EVAL_FREQ
INTERVAL=100000

def print_args(is_eval_experiment):
    if is_eval_experiment:
        file = open(EVAL_ARGS_OUTPUT_FILE, "w")
    else:
        file = open(ARGS_OUTPUT_FILE, "w")
    file.write("-----------------------------------------------\n")
    file.write("Nature Network: " + str(NATURE) + "\n")
    file.write("Use no-ops: " + str(USE_NO_OPS) + "\n")

    file.write("-----------------------------------------------\n")
    file.write("Algorithm parameters\n")
    file.write("-----------------------------------------------\n")
    file.write("Training frames: " + str(TRAINING_FRAMES) + "\n")
    file.write("Minibatch size: " + str(MINIBATCH_SIZE) + "\n")
    file.write("Replay capacity: " + str(REPLAY_CAPACITY) + "\n")
    file.write("History length: " + str(HIST_LEN) + "\n")
    file.write("Target network update frequency: " + str(TGT_UPDATE_FREQ) + "\n")
    file.write("Discount Factor: " + str(DISCOUNT) + "\n")
    file.write("Frame skip: " + str(ACT_REPEAT) + "\n")
    file.write("Backprop frequency: " + str(UPDATE_FREQ) + "\n")
    file.write("Learning rate: " + str(LEARNING_RATE) + "\n")
    file.write("Initial Exploration: " + str(INITIAL_EPSILON) + "\n")
    file.write("Final exploration: " + str(FINAL_EPSILON) + "\n")
    file.write("Exploration Duration: " + str(FINAL_EXPLORATION_FRAME) + "\n")
    file.write("Replay start size: " + str(REPLAY_START_SIZE) + "\n")
    file.write("No-op max: " + str(NO_OP_MAX) + "\n")
    file.write("Alpha (for RMSprop): " + str(ALPHA) + "\n")
    file.write("Min squared gradient: " + str(MIN_SQUARED_GRADIENT) + "\n")
    file.write("Test-time exploration: " + str(TEST_EPSILON) + "\n")
    file.write("Death ends episode:" + str(DEATH_ENDS_EPISODE) + "\n")
    file.write("-----------------------------------------------\n")
    file.write("\n")

    file.write("Random Seeds/Determinism\n")
    file.write("-----------------------------------------------\n")
    file.write("GPU Determinism: " + str(GPU_DETERMINISM) + "\n")
    file.write("Action repeat probability: " + str(REPEAT_ACTION_PROBABILITY) + "\n")
    file.write("Action repeat seed: " + str(REPEAT_ACTION_SEED) + "\n")
    file.write("ALE seed: " + str(ALE_SEED) + "\n")
    file.write("Network seed: " + str(NETWORK_SEED) + "\n")
    file.write("Exploration seed: " + str(EXPLORATION_SEED) + "\n")
    file.write("No-op seed: " + str(NO_OP_SEED) + "\n")
    file.write("Minibatch sampling seed: " + str(REPLAY_SAMPLE_SEED) + "\n")
    file.write("-----------------------------------------------\n")
    file.write("\n")

    file.write("Evaluation\n")
    file.write("-----------------------------------------------\n")
    file.write("Evaluation (whether to have an evaluation phase): " + str(EVALUATION_PHASE) + "\n")
    file.write("Evaluation Frequency: " + str(EVAL_FREQ) + "\n")
    file.write("Capped evaluation episodes: " + str(CAP_EVAL_EPISODES) + "\n")
    file.write("Maximum frames in eval episode: " + str(EVAL_MAX_FRAMES) + "\n")
    file.write("Number of Evaluation episodes: " + str(EVAL_EPISODES) + "\n")
    file.write("Evaluation Q Seed: " + str(EVAL_Q_SEED) + "\n")
    file.write("Evaluation No-op Seed: " + str(EVAL_NO_OP_SEED) + "\n")
    file.write("Evaluation Batch Seed: " + str(EVAL_BATCH_SEED) + "\n")
    file.write("Evaluation Exploration Seed: " + str(EVAL_EXP_SEED) + "\n")
    file.write("Evaluation Q No-op Seed:" + str(EVAL_Q_NO_OP_SEED) + "\n")
    file.write("-----------------------------------------------\n")
    file.write("\n")

    file.write("Files\n")
    file.write("-----------------------------------------------\n")
    file.write("Checkpoint directory: " + str(CHECKPOINT_DIR) + "\n")
    file.write("Arguments output file: " + str(ARGS_OUTPUT_FILE) + "\n")
    file.write("Evaluation arguments output file: " + str(EVAL_ARGS_OUTPUT_FILE) + "\n")
    file.write("-----------------------------------------------\n")
    file.write("\n")

    file.write("Other\n")
    file.write("-----------------------------------------------\n")
    file.write("Evaluation length: " + str(EVAL_STEPS) + "\n")
    file.write("Checkpoint frequency: " + str(CHECKPOINT_FREQUENCY) + "\n")
    file.write("Interval for storing weights " + str(INTERVAL) + "\n")
    file.write("-----------------------------------------------\n")
    file.close()
