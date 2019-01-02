import torch
import argparse
import numpy as np
from eval import DeterministicEvaluator
from train import train

def print_args(args, file):
	arguments = vars(args)
	for arg in arguments:
		file.write(str(arg) + ":" + str(arguments[arg]) + "\n")

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("--rom", type=str, required=True)

	##################################################
	##            Determinism parameters            ##
	##################################################
	'''
	An ALE stochasticity parameter that determines the
	probability that agent repeats its prior action. 
	Should be set to 0 for a deterministic implementation.
	'''
	parser.add_argument("--repeat-action-probability", type=float, default=0.0)
	parser.add_argument("--repeat-action-seed", type=int, default=123)

	'''
	Random seed for the Arcade Learning Environment
	'''
	parser.add_argument("--ale-seed", type=int, default=123)
	'''
	A flag to turn on Determinism in the GPU
	'''
	parser.add_argument("--gpu-determinism", type=bool, default=True)
	'''
	A random seed to seed a random number generator that 
	controls exploration
	'''
	parser.add_argument("--exploration-seed", type=int, default=123)
	'''
	A random seed to seed a random number generator for 
	exploration to control no-ops. A random number of 
	No-ops are performed at the beginning of each episode
	in order to randomize the initial start state.
	'''
	parser.add_argument("--no-op-seed", type=int, default=123)
	'''
	As with most deep learning algorithms, Deep Q-learning
	use random minibatch sampling. This seed is for a random
	number generator that controls the indices of the samples.
	'''
	parser.add_argument("--replay-sample-seed", type=int, default=123)
	'''
	A network seed for PyTorch. This allows us to control
	the weight initialization of the neural network.
	'''
	parser.add_argument("--network-seed", type=int, default=123)
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
	parser.add_argument("--nature", type=bool, default=True)
	##################################################
	##             Algorithm parameters             ##
	##################################################
	parser.add_argument("--training-frames", type=int, default=50000000)
	# 50000000 if NATURE else 10000000
	parser.add_argument("--minibatch-size", type=int, default=32)
	parser.add_argument("--replay-capacity", type=int, default=1000000)
	parser.add_argument("--hist-len", type=int, default=4)
	parser.add_argument("--tgt-update-freq", type=int, default=10000)
	parser.add_argument("--discount", type=float, default=0.99)
	parser.add_argument("--act-repeat", type=int, default=4)
	# Number of timesteps between gradient updates.
	# 4 if NATURE else 2
	parser.add_argument("--update-freq", type=int, default=4)
	parser.add_argument("--learning-rate", type=float, default=0.00025)
	# 0.00025 if NATURE else 1e-4
	# initial exploration epsilon
	parser.add_argument("--initial-epsilon", type=float, default=1.0)
	# the lowest epsilon used in the training process
	parser.add_argument("--final-epsilon", type=float, default=0.1)
	# 0.01 if REPEAT_ACTION_PROBABILITY > 0.0 else 0.1
	'''
	In the first FINAL_EXPLORATION_FRAME frames, we anneal
	the exploration epsilon from INITIAL_EPSILON to
	FINAL_EPSILON
	'''
	parser.add_argument("--final-exploration-frame", type=int, default=1000000)
	parser.add_argument("--replay-start-size", type=int, default=50000)
	'''
	The maximum number of no-ops at
	the beginning of an episode.
	'''
	parser.add_argument("--no-op-max", type=int, default=30)
	parser.add_argument("--alpha", type=float, default=0.95)
	parser.add_argument("--min-squared-gradient", type=float, default=0.01)
	# 0.01 if NATURE else 1e-6
	'''
	The exploration epsilon used at 
	test time.
	'''
	parser.add_argument("--test-epsilon", type=float, default=0.05)
	# 0.005 if REPEAT_ACTION_PROBABILITY > 0.0 else 0.05
	'''
	In some Atari games the agent has many lives.
	DEATH_ENDS_EPISODE specifies whether we should terminate
	an episode on the loss of a life during training.
	'''
	# True if NATURE else False
	parser.add_argument("--death-ends-episode", type=bool, default=True)

	##################################################
	##            Evaluation Parameters             ##
	##################################################
	parser.add_argument("--eval-freq", type=int, default=250000)
	#EVAL_FREQ = 250000 if NATURE else 100000
	parser.add_argument("--eval-steps", type=int, default=125000)
	parser.add_argument("--cap-eval-episodes", type=bool, default=True)
	parser.add_argument("--eval-max-frames", type=int, default=18000)
	parser.add_argument("--eval-episodes", type=int, default=100)
	parser.add_argument("--eval-batch-seed", type=int, default=123)
	parser.add_argument("--eval-exp-seed", type=int, default=123)
	parser.add_argument("--eval-no-op-seed", type=int, default=123)
	parser.add_argument("--eval-q-seed", type=int, default=123)
	parser.add_argument("--eval-q-no-op-seed", type=int, default=123)

	parser.add_argument("--eval-init-states-file", type=str,
						default="/home/ubuntu/repro_dqn/files/initstates.txt")

	##################################################
	##                   Files                      ##
	##################################################
	'''
	CHECKPOINT_DIR - directory to store the network checkpoints for reloading
	ARGS_OUTPUT_FILE - file to output the arguments from this file!
	EVAL_OUTPUT_FILE - file output the evaluations to
	'''
	parser.add_argument("--checkpoint-dir", type=str,
						default="/home/ubuntu/breakoutresults/checkpoints")
	parser.add_argument("--args-output-file", type=str,
						default="/home/ubuntu/breakoutresults/args.txt")
	parser.add_argument("--eval-output-file", type=str,
						default="/home/ubuntu/breakoutresults/evalargs.txt")
	##################################################
	##            Weight storage Parameters         ##
	##################################################
	parser.add_argument("--checkpoint-frequency", type=int,
						default=250000)
	#CHECKPOINT FREQUENCY SHOULD EQUAL EVAL FREQUENCY

	args = parser.parse_args()

	# SETUP
	'''
	Create random number generators for each of exploration,
	random no-ops, and minibatch sampling, all sources of 
	randomness within deep Q-learning.
	'''
	RNDEXP=np.random.RandomState(args.exploration_seed)
	RNDNO_OP=np.random.RandomState(args.no_op_seed)
	RND_MEMSAMPLE=np.random.RandomState(args.replay_sample_seed)
	RND_ACTREPEAT=np.random.RandomState(args.repeat_action_seed) 

	#Pytorch
	torch.manual_seed(args.network_seed)
	if torch.cuda.is_available():
	    print "Using cuda....."
	    torch.cuda.manual_seed_all(args.network_seed)
	    torch.backends.cudnn.deterministic = args.gpu_determinism
	else:
	    print "Not using cuda..."
	args_file = open(args.args_output_file, "w")
	print_args(args, args_file)

	evaluator = DeterministicEvaluator(args.eval_init_states_file,
		args.cap_eval_episodes,
		args.eval_max_frames,
		args.act_repeat,
		args.hist_len,
		args.rom,
		args.ale_seed,
		args.repeat_action_probability,
		args.eval_output_file)

	train(training_frames=args.training_frames,
		learning_rate=args.learning_rate,
		alpha=args.alpha,
		min_squared_gradient=args.min_squared_gradient,
		minibatch_size=args.minibatch_size,
		replay_capacity=args.replay_capacity, 
		hist_len=args.hist_len,
		tgt_update_freq=args.tgt_update_freq,
		discount=args.discount,
		act_rpt=args.act_repeat,
		upd_freq=args.update_freq, 
		init_epsilon=args.initial_epsilon,
		fin_epsilon=args.final_epsilon, 
		fin_exp=args.final_exploration_frame,
		replay_start_size=args.replay_start_size, 
		no_op_max=args.no_op_max,
		death_ends_episode=args.death_ends_episode,
		ale_seed=args.ale_seed,
		eval_freq=args.eval_freq,
		nature=args.nature,
		checkpoint_frequency=args.checkpoint_frequency,
		checkpoint_dir=args.checkpoint_dir,
		repeat_action_probability=args.repeat_action_probability,
		rnd_no_op=RNDNO_OP,
		rnd_exp=RNDEXP,
		rnd_act_repeat=RND_ACTREPEAT,
		rnd_buffer_sample=RND_MEMSAMPLE,
		rom=args.rom,
		evaluator=evaluator)
