from ale_python_interface import ALEInterface
from constants import *
from copy import deepcopy

class ALEInterfaceWrapper:
	def __init__(self, repeat_action_probability=REPEAT_ACTION_PROBABILITY, rng=RND_ACTREPEAT):
		self.internal_action_repeat_prob = repeat_action_probability
		print "repeat_action_probability is " + str(repeat_action_probability)
		self.prev_action = 0
		self.rng_source = rng
		self.rng = deepcopy(self.rng_source)
		self.ale = ALEInterface()
		'''
		This sets the probability from the default 0.25 to 0.
		It ensures deterministic actions.
		'''
		self.ale.setFloat('repeat_action_probability', 0.0)

	def getScreenRGB(self):
		return self.ale.getScreenRGB()
		
	def game_over(self):
		return self.ale.game_over()

	def reset_game(self):
		self.ale.reset_game()

	def lives(self):
		return self.ale.lives()

	def getMinimalActionSet(self):
		return self.ale.getMinimalActionSet()

	def setInt(self, key, value):
		self.ale.setInt(key, value)

	def setFloat(self, key, value):
		self.ale.setFloat(key, value)

	def loadROM(self, rom):
		self.ale.loadROM(rom)

	def reset_action_seed(self):
		self.rng = deepcopy(self.rng_source)

	def act(self, action):
		actual_action = action
		if self.internal_action_repeat_prob > 0:
			if self.rng.uniform(0,1) < self.internal_action_repeat_prob:
				actual_action = self.prev_action
		self.prev_action = actual_action
		return self.ale.act(actual_action)
