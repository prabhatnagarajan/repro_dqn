from collections import namedtuple
import numpy as np

Experience = namedtuple('Experience', 'state action reward new_state game_over') 

'''
A more efficient implementation of replay memory
to optimize for space.

This implementation of ReplayMemory was inspired by the following
sources:
 - devsisters/DQN-tensorflow: https://github.com/devsisters/DQN-tensorflow
 - Deepmind's implementation: https://github.com/deepmind/dqn
'''
class ReplayMemory:
	def __init__(self, replay_capacity, hist_len):
		# Initialize replay memory to capacity replay_capacity
		self.replay_capacity = replay_capacity
		self.hist_len = hist_len
		self.s = np.empty((replay_capacity, 84, 84), dtype=np.uint8)
		self.a = np.empty(replay_capacity, dtype=np.uint8)
		self.r = np.empty(replay_capacity)
		self.terminals = np.empty(replay_capacity, dtype=np.bool)
		self.episode_time = np.empty(replay_capacity, dtype=np.uint8)
		self.size = 0
		self.insertLoc = 0

	def add_item(self, frame, action, reward, episode_done, episode_timestep):
		# input a_t, r_t, f_t+1, episode done at t+1
		self.s[self.insertLoc, ...] = frame
		self.a[self.insertLoc] = action
		self.r[self.insertLoc] = reward
		self.terminals[self.insertLoc] = episode_done
		self.episode_time[self.insertLoc] = episode_timestep
		self.size = max(self.size, self.insertLoc + 1)
		self.insertLoc = (self.insertLoc + 1) % self.replay_capacity

	'''
	Assumes indices chosen properly
	'''
	def get_state(self, index):
		index = index % self.size
		if index >= self.hist_len - 1:
			return self.s[(index - (self.hist_len - 1)):(index + 1),...]
		else:
			return self.s[[(index - i) % self.size for i in reversed(range(hist_len))],...]

	def sample_minibatch(self, minibatch_size, random_state):
		indices = []
		while len(indices) < minibatch_size:
			rnd_index = random_state.randint(self.hist_len, self.size - 1)
			# check if there's buffer wraparound
			if rnd_index >= self.insertLoc and rnd_index - self.hist_len < self.insertLoc:
				continue
			# skip sequences that overlap with episode ends...
			if self.terminals[(rnd_index - self.hist_len):rnd_index].any():
				continue
			# Skip sequences where a new episode starts (but not terminal)
			if not all(t < t2 for t, t2 in zip(self.episode_time[(rnd_index - self.hist_len):rnd_index], 
										self.episode_time[(rnd_index - self.hist_len + 1):rnd_index])):
				continue
			indices.append(rnd_index)
		minibatch = [Experience(state=np.expand_dims(self.get_state(index - 1).astype(np.float32)/255.0, axis=0), 
			action=self.a[index], 
			reward=self.r[index], 
			new_state=np.expand_dims(self.get_state(index).astype(np.float32)/255.0, axis=0),
			game_over=self.terminals[index]) for index in indices]
		return minibatch

class State:
	def __init__(self, hist_len):
		# Initialize a 1 x hist_len x 84 x 84  state
		self.hist_len = hist_len
		self.state = np.zeros((1, hist_len, 84, 84), dtype=np.float32)
		'''
		index of next image, indicates that the first 84 x 84 
		image should be at (0, 0)
		'''
		self.insertLoc = 0

	def add_frame(self, img):
		self.state[0, self.insertLoc, ...] = img.astype(np.float32)/255.0
		# The index to insert at cycles from betweem 0, 1, 2, 3
		self.insertLoc = (self.insertLoc + 1) % self.hist_len

	def get_state(self):
		'''
		return the stacked four frames in the correct order
		Example: Suppose the state contains the following frames:
		[f4 f1 f2 f3]. The return value should be [f1 f2 f3 f4].
		Since the most recent frame inserted in this example was 
		f4 at index 0,  self.insertLoc equals 1. Thus, we 
		"roll" the image a single space to the left, resulting in 
		[f1 f2 f3 f4]
		'''
		return np.roll(self.state, 0 - self.insertLoc, axis=1)