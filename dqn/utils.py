import os.path
import torch

'''
checkpointing source:
https://blog.floydhub.com/checkpointing-tutorial-for-tensorflow-keras-and-pytorch/
'''
def save_checkpoint(state, epoch, checkpoint_dir):
	filename = checkpoint_dir + '/epoch' + str(epoch) + '.pth.tar'
	print "Saving checkpoint for epoch " + str(epoch) + " at " + filename + " ..."
	torch.save(state, filename)  # save checkpoint
	print "Saved checkpoint."

def checkpoint_exists(checkpoint_dir, epoch):
	return os.path.isfile(checkpoint_dir+'/epoch'+ str(epoch) + '.pth.tar')

def get_checkpoint(checkpoint_dir, epoch):
	resume_weights = checkpoint_dir + '/epoch' + str(epoch) + '.pth.tar'
	if torch.cuda.is_available():
		print "Attempting to load Cuda weights..."
		checkpoint = torch.load(resume_weights)
		print "Loaded weights."
	else:
		print "Attempting to load weights for CPU..."
		# Load GPU model on CPU
		checkpoint = torch.load(resume_weights,
								map_location=lambda storage,
								loc: storage)
		print "Loaded weights."
	return checkpoint
'''
Assumes that weights are checkpointed for the input epoch.
'''
def load_epoch_checkpoint(model, checkpoint_dir, epoch):
	resume_weights = checkpoint_dir + '/epoch' + str(epoch) + '.pth.tar'
	checkpoint = get_checkpoint(checkpoint_dir, epoch)
	model.load_state_dict(checkpoint['state_dict'])
	print("=> loaded checkpoint '{}' (trained for {} epochs)".format(resume_weights, checkpoint['epoch']))


def float_tensor(input):
	if torch.cuda.is_available():
		return torch.cuda.FloatTensor(input)
	else:
		return torch.FloatTensor(input)

def perform_no_ops(ale, no_op_max, preprocessor, state, random_state):
	#perform nullops
	num_no_ops = random_state.randint(1, no_op_max + 1)
	for _ in range(num_no_ops):
		ale.act(0)
		preprocessor.add(ale.getScreenRGB())
	if len(preprocessor.preprocess_stack) < 2:
		ale.act(0)
		preprocessor.add(ale.getScreenRGB())
	state.add_frame(preprocessor.preprocess())
