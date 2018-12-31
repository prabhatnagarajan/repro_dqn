from ale_wrapper import ALEInterfaceWrapper
from preprocess import Preprocessor
from replaybuffer import *
import numpy as np
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
		self.eval_output_file = open(eval_output_file, "w")

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
			self.eval_output_file.write("Episode " + str(episode_num) + " reward is " + str(episode_reward) + "\n")
			episode_rewards.append(episode_reward)
			episode_num += 1

		avg_reward = float(sum(episode_rewards))/float(len(sequences))
		self.eval_output_file.write("\n")
		self.eval_output_file.write("Greedy Evaluation:\n")
		self.eval_output_file.write("-------------------------------------------------------\n")
		self.eval_output_file.write("(Greedy)Epoch Number: " + str(epoch) + "\n")
		self.eval_output_file.write("(Greedy)Average Reward: " + str(avg_reward) + "\n")
		self.eval_output_file.write("-------------------------------------------------------\n")
		self.eval_output_file.write("\n")
		self.eval_output_file.flush()
		return avg_reward

class DeepmindEvaluator:
	def __init__(self):
		raise NotImplementError()