from pdb import set_trace
import torch.nn as nn
import torch.nn.functional as F

class NatureNet(nn.Module):
	def __init__(self, num_output_actions):
		super(NatureNet, self).__init__()
		self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
		self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
		self.fc1 = nn.Linear(64 * 7 * 7, 512)
		self.output = nn.Linear(512, num_output_actions)

	def forward(self, input):
		conv1_output = F.relu(self.conv1(input))
		conv2_output = F.relu(self.conv2(conv1_output))
		conv3_output = F.relu(self.conv3(conv2_output))
		fc1_output = F.relu(self.fc1(conv3_output.view(conv3_output.size(0), -1)))	
		output = self.output(fc1_output)
		return conv1_output, conv2_output, conv3_output, fc1_output, output

class NipsNet(nn.Module):
	def __init__(self, num_output_actions):
		super(NipsNet, self).__init__()
		self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
		self.fc1 = nn.Linear(2592, 256)
		self.output = nn.Linear(256, num_output_actions)

	def forward(self, input):
		conv1_output = F.relu(self.conv1(input))
		conv2_output = F.relu(self.conv2(conv1_output))
		fc1_output = F.relu(self.fc1(conv2_output.view(conv2_output.size(0), -1)))	
		output = self.output(fc1_output)
		return conv1_output, conv2_output, fc1_output, output