from ale_wrapper import ALEInterfaceWrapper
import os
import sys
from preprocess import Preprocessor
from replaybuffer import *
from dqn import DQN
from replaybuffer import *
from torch.autograd import Variable
import utils
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ast

def perform_action_sweep(ale, preprocessor, state):
	#perform a sweep through entire action set
	for action in ale.getMinimalActionSet().tolist():
		ale.act(action)
		preprocessor.add(ale.getScreenRGB())
	state.add_frame(preprocessor.preprocess())

class DeterministicEvaluator:

	def __init__(self, eval_file, cap_eval_episodes, eval_max_steps,
				action_repeat, hist_len, rom, ale_seed, action_repeat_prob,
				eval_output_file):
		self.eval_file = eval_file
		self.sequences = self.get_states(self.eval_file)
		self.cap_eval_episodes = cap_eval_episodes
		self.eval_max_steps = eval_max_steps
		self.action_repeat = action_repeat
		self.hist_len = hist_len
		self.rom = rom
		self.ale_seed = ale_seed
		self.action_repeat_prob = action_repeat_prob
		self.eval_output_file = eval_output_file

	def get_states(self, eval_file):
		sequences = []
		input_file = open(eval_file)
		lines = input_file.readlines()
		for line in lines:
			seq = ast.literal_eval(line)
			sequences.append(seq)
		assert len(sequences) == 100
		return sequences

	def evaluate(self, agent, epoch):
		ale = self.setup_eval_env(self.ale_seed, self.action_repeat_prob, self.rom)
		self.eval_greedy(ale, agent, epoch)

	def setup_eval_env(self, ale_seed, action_repeat_prob, rom):
		ale = ALEInterfaceWrapper()
		#Set the random seed for the ALE

		ale.setInt('random_seed', ale_seed)
		'''
		This sets the probability from the default 0.25 to 0.
		It ensures deterministic actions.
		'''
		ale.setFloat('repeat_action_probability', action_repeat_prob)
		# Load the ROM file
		ale.loadROM(rom)
		return ale

	def eval_greedy(self, ale, agent, epoch):
		sequences = self.sequences
		ale.reset_action_seed()
		episode_rewards = []
		episode_num = 0
		for sequence in sequences:
			ale.reset_game()
			preprocessor = Preprocessor()
			state = State(self.hist_len)
			episode_frames = 0
			episode_reward = 0
			print "Sequence is " + str(sequence)
			for i in range(len(sequence)):
				ale.act(sequence[i])
				preprocessor.add(ale.getScreenRGB())
				if (i + 1) % self.action_repeat == 0:
					state.add_frame(preprocessor.preprocess())
			count = 0
			lives = ale.lives()
			while not (ale.game_over() or (self.cap_eval_episodes and episode_frames > self.eval_max_steps)):
				action = agent.eGreedy_action(state.get_state(), 0.0, np.random.RandomState(4))
				reward = 0
				for i in range(self.action_repeat):
					reward += ale.act(action)
					preprocessor.add(ale.getScreenRGB())
					episode_frames += 1
				img = preprocessor.preprocess()
				state.add_frame(img)
				episode_reward += reward
				count+= 1
				if ale.lives() < lives:
					perform_action_sweep(ale, preprocessor, state)
					lives = ale.lives()
			print "Episode " + str(episode_num) + " reward is " + str(episode_reward)
			episode_rewards.append(episode_reward)
			episode_num += 1

		avg_reward = float(sum(episode_rewards))/float(len(sequences))
		print ""
		print "Greedy Evaluation:"
		print "-------------------------------------------------------"
		print "(Greedy)Epoch Number: " + str(epoch)
		print "(Greedy)Average Reward: " + str(avg_reward)
		print "-------------------------------------------------------"
		print ""
		return avg_reward

def log_eval(num_episodes, episodic_rewards, total_reward, epoch):
	print ""
	print "Evaluation:"
	print "-------------------------------------------------------"
	print "(Score)Epoch Number: " + str(epoch)
	print "(Score)Number of Episodes: " + str(num_episodes)
	print "(Score)Total Reward: " + str(total_reward)
	print "(Score)Standard Deviation: " + str(np.std(episodic_rewards))
	print "(Score)Mean Episodic Reward: " + str(np.mean(episodic_rewards))
	print "(Score)Rewards: " + str(episodic_rewards)
	print "-------------------------------------------------------"
	print ""

class DeepmindEvaluator:
	def __init__(self):
		raise NotImplementError()

# def evaluate(ale, agent, no_op_max, hist_len, act_rpt, frames, held_out_states):
# 	cum_reward = 0
# 	episode_rewards = []
# 	eval_episodes=100
# 	for _ in range(eval_episodes):
# 		ale.reset_game()
# 		preprocessor = Preprocessor()
# 		episode_reward = 0
# 		state = State(hist_len)
# 		utils.perform_no_ops(ale, no_op_max, preprocessor, state, rnd_no_op)
# 		episode_frames = 0
# 		episode_reward = 0
# 		while not (ale.game_over() or (CAP_EVAL_EPISODES and episode_frames > EVAL_MAX_FRAMES)):
# 			action = agent.eGreedy_action(state.get_state(), TEST_EPSILON)
# 			reward = 0
# 			for i in range(act_rpt):
# 				reward += ale.act(action)
# 				preprocessor.add(ale.getScreenRGB())
# 				episode_frames += 1
# 			img = preprocessor.preprocess()
# 			state.add_frame(img)
# 			episode_reward += reward
				
# 		cum_reward += episode_reward
# 		episode_rewards.append(episode_reward)
# 	#compute average maximum q value on a set of held out states
# 	total_max_q = 0.0
# 	if len(held_out_states) > 0:
# 		for experience in held_out_states:
# 			net_outputs = agent.prediction_net(Variable(utils.float_tensor(experience.state)))
# 			q_vals = net_outputs[len(net_outputs) - 1].data.cpu().numpy()
# 			total_max_q += np.amax(q_vals)
# 		avg_max_q = total_max_q/float(len(held_out_states))
# 	else:
# 		# just set to 0 by default
# 		avg_max_q = 0
# 	log_eval(EVAL_EPISODES, episode_rewards, cum_reward, frames, avg_max_q)

# if __name__=='__main__':

# 	print_args(True)
# 	ale = ALEInterfaceWrapper()
# 	#Set the random seed for the ALE
# 	ale.setInt('random_seed', ALE_SEED)
# 	'''
# 	This sets the probability from the default 0.25 to 0.
# 	It ensures deterministic actions.
# 	'''
# 	ale.setFloat('repeat_action_probability', REPEAT_ACTION_PROBABILITY)
# 	# Load the ROM file
# 	ale.loadROM(sys.argv[1])

# 	agent = DQN(ale.getMinimalActionSet().tolist(), TEST_EPSILON, HIST_LEN, DISCOUNT, rom_name(sys.argv[1]))

# 	full_q_eval()

# 	full_greedy_eval()

