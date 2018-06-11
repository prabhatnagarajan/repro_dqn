from constants import *
from cnn import NipsNet, NatureNet
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

def perform_no_ops(ale, preprocessor, state, random_state):
	#perform nullops
	if USE_NO_OPS:
		num_no_ops = random_state.randint(1, NO_OP_MAX + 1)
		for _ in range(num_no_ops):
			ale.act(0)
			preprocessor.add(ale.getScreenRGB())
	if len(preprocessor.preprocess_stack) < 2:
		ale.act(0)
		preprocessor.add(ale.getScreenRGB())
	state.add_frame(preprocessor.preprocess())

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

def get_held_out_states(ale, agent):
	EVAL_Q_EXP = np.random.RandomState(EVAL_Q_SEED)
	Q_NO_OP = np.random.RandomState(EVAL_Q_NO_OP_SEED)
	num_frames = 0
	replay_memory = ReplayMemory(REPLAY_CAPACITY, HIST_LEN)
	while num_frames < REPLAY_START_SIZE:
		ale.reset_game()
		state = State(HIST_LEN)
		preprocessor = Preprocessor()
		perform_no_ops(ale, preprocessor, state, Q_NO_OP)
		lives = ale.lives()
		episode_done = False
		while not episode_done:
			action = agent.eGreedy_action(state.get_state(), 1.0, random_state=EVAL_Q_EXP)
			for i in range(ACT_REPEAT):
				ale.act(action)
				preprocessor.add(ale.getScreenRGB())
			img = preprocessor.preprocess()
			episode_done = ale.game_over() or (ale.lives() < lives and DEATH_ENDS_EPISODE)
			replay_memory.add_item(img, action, 0, episode_done)
			num_frames += 1
	EVAL_MINIBATCH_RNG = np.random.RandomState(EVAL_BATCH_SEED)
	return replay_memory.sample_minibatch(500, random_state=EVAL_MINIBATCH_RNG)

def evaluate_q(ale, agent, epoch, held_out_states):
	total_max_q = 0.0
	for experience in held_out_states:
		net_outputs = agent.prediction_net(Variable(utils.float_tensor(experience.state)))
		q_vals = net_outputs[len(net_outputs) - 1].data.cpu().numpy()
		total_max_q += np.amax(q_vals)
	avg_max_q = total_max_q/float(len(held_out_states))
	print ""
	print "Q-Evaluation:"
	print "-------------------------------------------------------"
	print "(Q)Epoch Number: " + str(epoch)
	print "(Q)Average Maximum Q-value: " + str(avg_max_q)
	print "-------------------------------------------------------"
	print ""
	return avg_max_q

def eval_greedy(ale, agent, sequences, epoch):
	ale.reset_action_seed()
	episode_rewards = []
	episode_num = 0
	for sequence in sequences:
		ale.reset_game()
		preprocessor = Preprocessor()
		state = State(HIST_LEN)
		episode_frames = 0
		episode_reward = 0
		print "Sequence is " + str(sequence)
		for i in range(len(sequence)):
			ale.act(sequence[i])
			preprocessor.add(ale.getScreenRGB())
			if (i + 1) % ACT_REPEAT == 0:
				state.add_frame(preprocessor.preprocess())
		count = 0
		lives= ale.lives()
		while not (ale.game_over() or (CAP_EVAL_EPISODES and episode_frames > EVAL_MAX_FRAMES)):
			action = agent.eGreedy_action(state.get_state(), 0.0)
			reward = 0
			for i in range(ACT_REPEAT):
				reward += ale.act(action)
				preprocessor.add(ale.getScreenRGB())
				episode_frames += 1
			img = preprocessor.preprocess()
			state.add_frame(img)
			episode_reward += reward
			count+= 1
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

def rom_name(path):
	return os.path.splitext(os.path.basename(path))[0]
	
def get_states():
	sequences = []
	input_file = open(EVAL_INIT_STATES_FILE)
	lines = input_file.readlines()
	for line in lines:
		seq = ast.literal_eval(line)
		sequences.append(seq)
	assert len(sequences) == 100
	return sequences

def full_q_eval():
	print "Getting the held out states..."
	held_out_states = get_held_out_states(ale, agent)
	print "Beginning Q evaluations"
	avg_q = []
	epoch  = 0
	while (utils.checkpoint_exists(agent.checkpoint_directory, epoch)):
		utils.load_epoch_checkpoint(agent.prediction_net, agent.checkpoint_directory, epoch)
		avg_q.append(evaluate_q(ale, agent, epoch, held_out_states))
		epoch += 1

def full_greedy_eval():
	print "Beginning Greedy evaluations"
	sequences = get_states()
	avg_rewards = []
	epoch  = 0
	while (utils.checkpoint_exists(agent.checkpoint_directory, epoch)):
		utils.load_epoch_checkpoint(agent.prediction_net, agent.checkpoint_directory, epoch)
		avg_rewards.append(eval_greedy(ale, agent, sequences, epoch))
		epoch += 1	

if __name__=='__main__':

	print_args(True)
	ale = ALEInterfaceWrapper()
	#Set the random seed for the ALE
	ale.setInt('random_seed', ALE_SEED)
	'''
	This sets the probability from the default 0.25 to 0.
	It ensures deterministic actions.
	'''
	ale.setFloat('repeat_action_probability', REPEAT_ACTION_PROBABILITY)
	# Load the ROM file
	ale.loadROM(sys.argv[1])

	agent = DQN(ale.getMinimalActionSet().tolist(), TEST_EPSILON, HIST_LEN, DISCOUNT, rom_name(sys.argv[1]))

	full_q_eval()

	full_greedy_eval()

