import sys
import os
from dqn import DQN
from ale_wrapper import ALEInterfaceWrapper
from collections import deque
from collections import namedtuple
from preprocess import Preprocessor
import numpy as np
from replaybuffer import *
from torch.autograd import Variable
import utils

def train(training_frames,
		minibatch_size,
		replay_capacity, 
		hist_len,
		tgt_update_freq,
		discount,
		act_rpt,
		upd_freq, 
		init_epsilon,
		fin_epsilon, 
		fin_exp,
		replay_start_size, 
		no_op_max,
		death_ends_episode,
		ale_seed,
		eval_freq,
		checkpoint_frequency,
		rnd_no_op,
		evaluator):
	#Create ALE object
	if len(sys.argv) < 2:
	  print 'Usage:', sys.argv[0], 'rom_file'
	  sys.exit()  
	ale = ALEInterfaceWrapper()

	#Set the random seed for the ALE
	ale.setInt('random_seed', ale_seed)

	# Load the ROM file
	ale.loadROM(sys.argv[1])

	#initialize epsilon
	epsilon = init_epsilon
	# How much epsilon decreases at each time step. The annealing amount.
	epsilon_delta = (init_epsilon - fin_epsilon)/fin_exp

	print "Minimal Action set is:"
	print ale.getMinimalActionSet()


	''' 
	store random states for Q evaluation (not core to the algorithm).
	Can safely ignore.
	'''
	random_states_memory = []
	# create DQN agent
	agent = DQN(ale.getMinimalActionSet().tolist(), epsilon, hist_len, discount, rom_name(sys.argv[1]))
	# Initialize replay memory to capacity replay_capacity
	replay_memory = ReplayMemory(replay_capacity, hist_len)
	num_frames = 0 #same as time step
	episode_num = 1
	# Main training loop
	while num_frames < training_frames:
		# create a state variable of size hist_len
		state = State(hist_len)
		preprocessor = Preprocessor()
		# perform a random number of no ops to start the episode
		utils.perform_no_ops(ale, no_op_max, preprocessor, state, rnd_no_op)
		# total episode reward is 0
		total_reward = 0
		lives = ale.lives()
		episode_done = False
		# episode loop
		while not episode_done:
			if num_frames % checkpoint_frequency == 0:
				epoch = num_frames/checkpoint_frequency
				agent.checkpoint_network(epoch)

			action = agent.get_action(state.get_state())

			reward = 0
			#skip frames by repeating action
			for i in range(act_rpt):
				reward = reward + ale.act(action)
				#add the images on stack 
				preprocessor.add(ale.getScreenRGB())

			#increment episode reward before clipping the reward for training
			total_reward += reward
			reward = np.clip(reward, -1, 1)

			# get the preprocessed new frame
			img = preprocessor.preprocess()
			state.add_frame(img)

			episode_done = ale.game_over() or (ale.lives() < lives and DEATH_ENDS_EPISODE)
			#store transition
			replay_memory.add_item(img, action, reward, episode_done)

			'''
			Training. We only train once buffer has filled to 
			size=replay_start_size
			'''
			if (num_frames > replay_start_size):
				# not core to the algorithm, set aside 500 early states.
				if len(random_states_memory) == 0:
					random_states_memory = replay_memory.sample_minibatch(500)
					evaluate(ale, agent, no_op_max, hist_len, act_rpt, 0, random_states_memory)
				# anneal epsilon.
				epsilon = max(epsilon - epsilon_delta, fin_epsilon)
				agent.set_epsilon(epsilon)
				if num_frames % eval_freq == 0:
					evaluate(ale, agent, no_op_max, hist_len, act_rpt, num_frames, random_states_memory)
				if num_frames % upd_freq == 0:
					agent.train(replay_memory, minibatch_size) 
			num_frames = num_frames + 1
			'''
			Inconsistency in Deepmind code versus Paper. In code they update target
			network every tgt_update_freq actions. In the the paper they say to do
			it every tgt_update_freq parameter updates.
			'''
			if num_frames % tgt_update_freq == 1:
				print "Copying Network..."
				agent.copy_network()
				print "Done Copying."
		   

		log(episode_num, total_reward, num_frames)
		# if game is not over, then continue with new life
		if ale.game_over():
			ale.reset_game()
		episode_num = episode_num + 1

	if num_frames == training_frames:
		evaluate(ale, agent, no_op_max, hist_len, act_rpt, num_frames, random_states_memory)
		agent.checkpoint_network(training/checkpoint_frequency)
	print "Number " + str(num_frames)

def log(episode_num, reward, frames):
	print ""
	print "-------------------------------------------------------"
	print "Episodes: " + str(episode_num)
	print "Reward: " + str(reward)
	print "Frames (excluding frame skip): " + str(frames)
	print "-------------------------------------------------------"
	print ""

def log_eval(num_episodes, episodic_rewards, total_reward, num_frames, avg_max_q):
	print ""
	print "Evaluation:"
	print "-------------------------------------------------------"
	print "Epoch Number: " + str(int(num_frames/eval_freq))
	print "Average Maximum Q-value: " + str(avg_max_q)
	print "Number of Episodes: " + str(num_episodes)
	print "Total Reward: " + str(total_reward)
	mean = float(total_reward)/float(max(1, num_episodes))
	print "Mean Reward: " + str(mean)
	if num_episodes > 0:
		print "Standard Deviation: " + str(np.std(episodic_rewards))
		print "Mean Episodic Reward: " + str(np.mean(episodic_rewards))
		print "Rewards: " + str(episodic_rewards)
	print "-------------------------------------------------------"
	print ""

def rom_name(path):
	return os.path.splitext(os.path.basename(path))[0]

if __name__ == '__main__':
	print_args(False)
	train()