import os
from dqn import DQN
from ale_wrapper import ALEInterfaceWrapper
from preprocess import Preprocessor
import numpy as np
from replaybuffer import *
import utils

def train(training_frames,
		learning_rate,
		alpha,
		min_squared_gradient,
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
		nature,
		checkpoint_frequency,
		checkpoint_dir,
		repeat_action_probability,
		rnd_no_op,
		rnd_exp,
		rnd_act_repeat,
		rnd_buffer_sample,
		rom,
		evaluator):

	#Create ALE object
	ale = ALEInterfaceWrapper(repeat_action_probability, rnd_act_repeat)

	#Set the random seed for the ALE
	ale.setInt('random_seed', ale_seed)

	# Load the ROM file
	ale.loadROM(rom)

	#initialize epsilon
	epsilon = init_epsilon
	# How much epsilon decreases at each time step. The annealing amount.
	epsilon_delta = (init_epsilon - fin_epsilon)/fin_exp

	print "Minimal Action set is:"
	print ale.getMinimalActionSet()

	# create DQN agent
	agent = DQN(ale.getMinimalActionSet().tolist(),
				learning_rate,
				alpha,
				min_squared_gradient,
				nature,
				checkpoint_frequency,
				checkpoint_dir,
				epsilon,
				hist_len,
				discount,
				rom_name(rom),
				rnd_exp,
				rnd_buffer_sample)
	# Initial evaluation
	evaluator.evaluate(agent, 0)
	# Initialize replay memory to capacity replay_capacity
	replay_memory = ReplayMemory(replay_capacity, hist_len)
	timestep = 0
	episode_num = 1
	# Main training loop
	while timestep < training_frames:
		# create a state variable of size hist_len
		state = State(hist_len)
		preprocessor = Preprocessor()
		# perform a random number of no ops to start the episode
		utils.perform_no_ops(ale, no_op_max, preprocessor, state, rnd_no_op)
		# total episode reward is 0
		total_reward = 0
		lives = ale.lives()
		episode_done = False
		time_since_term = 0
		# episode loop
		while not episode_done:
			if timestep % checkpoint_frequency == 0:
				epoch = timestep/checkpoint_frequency
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
			replay_memory.add_item(img, action, reward, episode_done, time_since_term)

			'''
			Training. We only train once buffer has filled to 
			size=replay_start_size
			'''
			if (timestep > replay_start_size):
				# anneal epsilon.
				epsilon = max(epsilon - epsilon_delta, fin_epsilon)
				agent.set_epsilon(epsilon)
				if timestep % eval_freq == 0:
					evaluator.evaluate(agent, timestep/eval_freq)
					ale.reset_game()
					# Break loop and start new episode after eval
					# Can help prevent getting stuck in episodes
					episode_done = True
				if timestep % upd_freq == 0:
					agent.train(replay_memory, minibatch_size) 
			timestep = timestep + 1
			time_since_term += 1
			'''
			Inconsistency in Deepmind code versus Paper. In code they update target
			network every tgt_update_freq actions. In the the paper they say to do
			it every tgt_update_freq parameter updates.
			'''
			if timestep % tgt_update_freq == 1:
				print "Copying Network..."
				agent.copy_network()
				print "Done Copying."
		   

		log(episode_num, total_reward, timestep)
		# if game is not over, then continue with new life
		if ale.game_over():
			ale.reset_game()
		episode_num = episode_num + 1

	if timestep == training_frames:
		evaluator.evaluate(agent, training_frames/eval_freq)
		agent.checkpoint_network(training/checkpoint_frequency)
	print "Number " + str(timestep)

def log(episode_num, reward, frames):
	print ""
	print "-------------------------------------------------------"
	print "Episodes: " + str(episode_num)
	print "Reward: " + str(reward)
	print "Frames (excluding frame skip): " + str(frames)
	print "-------------------------------------------------------"
	print ""

def rom_name(path):
	return os.path.splitext(os.path.basename(path))[0]

if __name__ == '__main__':
	print_args(False)
	train()