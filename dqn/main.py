import torch

if __name__ == '__main__':
	print_args(False)
	parser = argsparse.ArgumentParser()
	parser.add_argument()

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
	parser.add_argument("--use-no-ops", type=bool, default=True)
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

	train()