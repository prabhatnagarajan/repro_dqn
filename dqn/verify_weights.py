from pdb import set_trace
import numpy as np
import os
import constants
import utils
from cnn import NatureNet, NipsNet
import torch
import argparse


CHECKPOINT_FOLDERS = [
"/home/ubuntu/breakoutresults/checkpoints/breakout",
"/home/ubuntu/breakoutresults/checkpoints/breakout"]

def checkpoints_all_exist(epoch):
	for checkpoint_dir in CHECKPOINT_FOLDERS:
		if not utils.checkpoint_exists(checkpoint_dir, epoch):
			return False
	return True

def log_comparison(nature,
					conv1,
					conv2,
					conv3,
					fc,
					output,
					bias1,
					bias2,
					bias3,
					fc_bias,
					output_bias):
	print "Conv1 weights are same: " + str(conv1)
	print "Conv2 weights are same: " + str(conv2)
	if nature:
		print "Conv3 weights are same: " + str(conv3)
	print "Bias1 weights are same: " + str(bias1)
	print "Bias2 weights are same: " + str(bias2)
	if nature:
		print "Bias3 weights are same: " + str(bias3)
	print "FC1 weights are same: " + str(fc)
	print "FC1 bias weights are same: " + str(fc_bias)
	print "Output weights are same: " + str(output)
	print "Output bias weights are same: " + str(output_bias)
	print "-------------------------------------------------------------------"

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--nature", default=True, type=bool)
	args = parser.parse_args()

	epoch=0
	everywhere_all_the_time = True
	while checkpoints_all_exist(epoch):
		nets = []
		for checkpoint_dir in CHECKPOINT_FOLDERS:
			checkpoint = utils.get_checkpoint(checkpoint_dir, epoch)
			nets.append(checkpoint['state_dict'])

		'''
		Print results
		'''
		print "EPOCH: " + str(epoch)
		conv1 = True
		conv2 = True
		fc1 = True
		output = True
		conv1_bias = True
		conv2_bias = True
		fc1_bias = True
		output_bias = True
		if args.nature:
			conv3 = True
			conv3_bias = True
		print "Comparing nets..."
		for i in range(len(nets) - 1):
			net_a = nets[i]
			net_b = nets[i + 1]
			conv1 = conv1 and torch.equal(net_a['conv1.weight'], net_b['conv1.weight'])
			conv2 = conv1 and torch.equal(net_a['conv2.weight'], net_b['conv2.weight'])
			fc1 = fc1 and torch.equal(net_a['fc1.weight'], net_b['fc1.weight'])
			output = output and torch.equal(net_a['output.weight'], net_b['output.weight'])
			conv1_bias = conv1_bias and torch.equal(net_a['conv1.bias'], net_b['conv1.bias'])
			conv2_bias = conv2_bias and torch.equal(net_a['conv2.bias'], net_b['conv2.bias'])
			fc1_bias = fc1_bias and torch.equal(net_a['fc1.bias'], net_b['fc1.bias'])
			output_bias = output_bias and torch.equal(net_a['output.bias'], net_b['output.bias'])
			if args.nature:
				conv3 = conv3 and torch.equal(net_a['conv3.weight'], net_b['conv3.weight'])
				conv3_bias = conv3_bias and torch.equal(net_a['conv3.bias'], net_b['conv3.bias'])
			else:
				conv3 = None
				conv3_bias = None
		if args.nature:
			everywhere_all_the_time = everywhere_all_the_time and conv1 and conv2 and conv3 \
			and conv1_bias and conv2_bias and conv3_bias and fc1 and fc1_bias \
			and output and output_bias
		else:
			everywhere_all_the_time = everywhere_all_the_time and conv1 and conv2 \
			and conv1_bias and conv2_bias and fc1 and fc1_bias and output and output_bias			

		log_comparison(args.nature,
			conv1,
			conv2,
			conv3,
			fc1,
			output,
			conv1_bias,
			conv2_bias,
			conv3_bias,
			fc1_bias,
			output_bias)
		epoch += 1
	if epoch == 0:
		print "No networks to compare..."
	else:
		print "The networks (all of them) are equal everywhere (all layers), all the time (every epoch): " \
	 	+ str(everywhere_all_the_time)
