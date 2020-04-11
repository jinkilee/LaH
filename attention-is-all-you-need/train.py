import sys
sys.path.insert(0, '../modeling')

import os
import dill
import torch
import random
import glob
import setting
import numpy as np
import sentencepiece as spm
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
import logging
import logging.config

#from visdom import Visdom
from tqdm import tqdm, trange
from loss_compute import SimpleLossCompute, MultiGPULossCompute, MultiGPULossCompute_
from utils import fix_torch_randomness, get_sentencepiece, get_number_of_params
from transformer import make_model 
from dataset import load_dataset_aihub, KRENDataset, KRENField, MyIterator, batch_size_fn, rebatch
#from torch.nn.parallel import DistributedDataParallel as DDP
from apex.parallel import DistributedDataParallel as DDP
from torch.autograd import Variable
from setting import *
from feature import *
from optimizer import NoamOpt, get_std_opt
from regular import LabelSmoothing
from apex import amp

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
torch.backends.cudnn.benchmark = True

# define logger 
logging.config.fileConfig('logging.conf')
log = logging.getLogger('LaH')

fileHandler = logging.FileHandler('train.log')
log.addHandler(fileHandler)

# define ENV variables
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '10002'

# FIXME: fix logging function
print_log = log.debug

def save_model(args, model, optimizer, epoch, loss, model_name='model-tmp.bin'):
	model_full_path = os.path.join(args.model_path, model_name)

	# FIXME: fix model name
	torch.save({
		'epoch': epoch + 1,					  # need only for retraining
		'state_dict': model.module.state_dict(),
		'best_val_loss': loss,		  # need only for retraining
		'optimizer' : optimizer.optimizer.state_dict(), # need only for retraining
		'learning_rate' : optimizer._rate, # need only for retraining
	}, model_full_path)

	sum_of_weight = sum([p[1].data.sum() for p in model.named_parameters()])
	print_log('model was saved at {} -> {:.4f}'.format(model_full_path, sum_of_weight))

def average_gradients(model):
	size = float(dist.get_world_size())
	for param in model.parameters():
		dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
		param.grad.data /= size

'''
print(src.shape, trg[:, :-1].shape, src_mask.shape, trg_mask[:,:-1,:-1].shape)
torch.Size([285, 7]) torch.Size([285, 10]) torch.Size([285, 1, 7]) torch.Size([285, 10, 10])
torch.Size([307, 9]) torch.Size([307, 9]) torch.Size([307, 1, 9]) torch.Size([307, 9, 9])
torch.Size([160, 13]) torch.Size([160, 21]) torch.Size([160, 1, 13]) torch.Size([160, 21, 21])
torch.Size([222, 12]) torch.Size([222, 14]) torch.Size([222, 1, 12]) torch.Size([222, 14, 14])
torch.Size([160, 13]) torch.Size([160, 21]) torch.Size([160, 1, 13]) torch.Size([160, 21, 21])
'''
def train_epoch(train_iter, model, criterion, model_opt, rank, epoch, fp16):

	"Standard Training and Logging Function"
	losses = []
	total_loss = 0.0
	total_count = 0.0
	model.train()
	with tqdm(train_iter, desc='training {}th epoch'.format(epoch)) as tbar:
		for i, batch in enumerate(tbar):
			src, trg, src_mask, trg_mask = \
				batch.src, batch.trg, batch.src_mask, batch.trg_mask

			out = model.forward(src, trg[:, :-1], src_mask, trg_mask[:, :-1, :-1])
			loss = loss_backprop(
				generator=model.module.generator, 
				criterion=criterion, 
				optimizer=model_opt,
				out=out, 
				targets=trg[:, 1:], 
				normalize=batch.ntokens,
				fp16=fp16)
							
			model_opt.step()
			model_opt.zero_grad()
			tbar.set_postfix(loss=loss)
			if i % n_save == 1:
				losses.append(total_loss/total_count)
				total_loss = 0
				total_count = 0
			total_loss += loss
			total_count += 1

	return losses

def valid_epoch(valid_iter, model, criterion, model_opt, epoch, fp16):
	"Standard validation function"
	model.eval()
	total = 0.0
	total_tokens = 0.0
	with tqdm(valid_iter, desc='validating {}th epoch'.format(epoch)) as tbar:
		for batch in tbar:
			src, trg, src_mask, trg_mask = \
				batch.src, batch.trg, batch.src_mask, batch.trg_mask

			out = model.forward(src, trg[:, :-1], src_mask, trg_mask[:, :-1, :-1])
			total += loss_backprop(
				generator=model.module.generator, 
				criterion=criterion, 
				optimizer=model_opt,
				out=out, 
				targets=trg[:, 1:], 
				normalize=batch.ntokens, 
				fp16=fp16,
				bp=False) * batch.ntokens
			total_tokens += batch.ntokens
	return (total / total_tokens.float()).item()

def loss_backprop(generator, criterion, optimizer, out, targets, normalize, fp16, bp=True):
	"""
	Memory optmization. Compute each timestep separately and sum grads.
	"""

	assert out.size(1) == targets.size(1)
	total = 0.0
	out_grad = []
	for i in range(out.size(1)):
		out_column = Variable(out[:, i].data, requires_grad=True)
		gen = generator(out_column)
		loss = criterion(gen, targets[:, i]).float() / normalize.float()
		total += loss.item()

		if fp16:
			with amp.scale_loss(loss, optimizer) as scaled_loss:
				scaled_loss.backward()
		else:
			loss.backward()
		out_grad.append(out_column.grad.data.clone())
	if bp:
		out_grad = torch.stack(out_grad, dim=1)
		out.backward(gradient=out_grad)
	return total

parser = argparse.ArgumentParser()
parser.add_argument('--world_size', default=setting.world_size, type=int)
parser.add_argument('--multi_gpu', default=setting.multi_gpu, type=bool)
parser.add_argument('--epochs', default=setting.epochs, type=int)
parser.add_argument('--N', default=setting.N, type=int)
parser.add_argument('--d_model', default=setting.d_model, type=int)
parser.add_argument('--d_ff', default=setting.d_ff, type=int)
parser.add_argument('--h', default=setting.h, type=int)
parser.add_argument('--dropout', default=setting.dropout, type=float)
parser.add_argument('--train_batch_size', default=setting.train_batch_size, type=int)
parser.add_argument('--valid_batch_size', default=setting.valid_batch_size, type=int)
parser.add_argument('--model_path', default=setting.model_path, type=str)
parser.add_argument('--local_rank', type=int)
parser.add_argument('--fp16', default=setting.fp16, type=bool)
parser.add_argument('--opt-level', type=str)
parser.add_argument('--keep-batchnorm-fp32', type=str, default=setting.keep_batchnorm_fp32)
parser.add_argument('--loss-scale', type=str, default=None)


def main():
	args = parser.parse_args()

	if args.multi_gpu:
		ngpus_per_node = torch.cuda.device_count()
	else:
		ngpus_per_node = 1

	args.world_size = ngpus_per_node

	global best_acc1
	args.gpu = args.local_rank
	torch.cuda.set_device(args.gpu)

	dist.init_process_group(backend='nccl', 
							init_method='env://',
							world_size=args.world_size,
							rank=args.gpu)

	# load dataset
	#sent_pairs = load_dataset_aihub(path='data/')
	sent_pairs = load_dataset_aihub()
	print_log('GPU#{} seeding with {}'.format(args.gpu, args.gpu))

	# make dataloader with dataset
	# FIXME: RuntimeError: Internal: unk is not defined.
	inp_lang, out_lang = get_sentencepiece(src_prefix, trg_prefix)
	print_log('loaded input sentencepiece model: {}'.format(src_prefix))
	print_log('loaded output sentencepiece model: {}'.format(trg_prefix))

	# split train/valid sentence pairs
	n_train = int(len(sent_pairs) * 0.8)
	n_split = int(n_train * 1./args.world_size)
	print_log(n_split*args.gpu, n_split*(args.gpu+1))

	train_sent_pairs = sent_pairs[:n_train]
	print_log('train_sent_pairs before split: {}'.format(len(train_sent_pairs)))

	# split train datset by GPU
	train_sent_pairs = train_sent_pairs[n_split*args.gpu:n_split*(args.gpu+1)]
	train_sent_pairs = sorted(train_sent_pairs, key=lambda x: (len(x[0]), len(x[1])))
	print_log('train_sent_pairs after split: {} --> GPU:{}'.format(len(train_sent_pairs), args.gpu))

	valid_sent_pairs = sent_pairs[n_train:]
	print_log('valid_sent_pairs: {}'.format(len(valid_sent_pairs)))

	# these are used for defining tokenize method and some reserved words
	SRC = KRENField(pad_token='<pad>')
	TRG = KRENField(pad_token='<pad>')

	SRC.decode = inp_lang.DecodeIds
	TRG.decode = out_lang.DecodeIds
	SRC.encode = inp_lang.EncodeAsIds
	TRG.encode = out_lang.EncodeAsIds

	# load SRC/TRG
	if not os.path.exists('spm/{}.model'.format(src_prefix)) or \
		not os.path.exists('spm/{}.model'.format(trg_prefix)):
		# build vocabulary
		SRC.build_vocab(train.src)
		TRG.build_vocab(train.trg)
		torch.save(SRC.vocab, 'spm/{}.spm'.format(src_prefix), pickle_module=dill)
		torch.save(TRG.vocab, 'spm/{}.spm'.format(trg_prefix), pickle_module=dill)
		print_log('input vocab was created and saved: spm/{}.spm'.format(src_prefix))
		print_log('output vocab was created and saved: spm/{}.spm'.format(trg_prefix))
	else:
		src_vocab = torch.load('spm/{}.spm'.format(src_prefix), pickle_module=dill)
		trg_vocab = torch.load('spm/{}.spm'.format(trg_prefix), pickle_module=dill)
		SRC.vocab = src_vocab
		TRG.vocab = trg_vocab
		print_log('input vocab was loaded: spm/{}.spm'.format(src_prefix))
		print_log('output vocab was loaded: spm/{}.spm'.format(trg_prefix))

	# make dataloader from KRENDataset
	train, valid, test = KRENDataset.splits(sent_pairs, (SRC, TRG), inp_lang, out_lang, encoding_type='pieces')

	# output -> ['<s>', '▁', 'Central', '▁Asian', '▁c', 'u', 'is', ... '▁yesterday', '.', '</s>']
	train_iter = MyIterator(train, batch_size=args.train_batch_size, device=0,
							repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
							batch_size_fn=batch_size_fn, train=True)
	valid_iter = MyIterator(valid, batch_size=args.valid_batch_size, device=0,
							repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
							batch_size_fn=batch_size_fn, train=False)
	# fix torch randomness
	fix_torch_randomness()

	# define input/output size
	args.inp_n_words = src_vocab_size
	args.out_n_words = trg_vocab_size
	print_log('inp_n_words: {} out_n_words: {}'.format(args.inp_n_words, args.out_n_words))

	# define model
	model = make_model(
		args.inp_n_words, 
		args.out_n_words,
		N=N,
		d_model=args.d_model,
		d_ff=args.d_ff,
		h=args.h,
		dropout=args.dropout)
	print_log('number of model parameters: {}'.format(get_number_of_params(model)))
	model.cuda()
	optimizer = get_std_opt(model, args.fp16)

	# initizlie model and optimizer for amp
	model, optimizer = amp.initialize(
		model,
		optimizer,
		opt_level=args.opt_level,
		#keep_batchnorm_fp32=args.keep_batchnorm_fp32,
		#loss_scale=args.loss_scale
	)
	#optimizer.optimizer = opt

	if args.fp16:
		model = DDP(model, delay_allreduce=True)
	else:
		model = DDP(model, device_ids=[args.gpu])

	# define model
	criterion = LabelSmoothing(size=args.out_n_words, padding_idx=0, smoothing=0.1)
	criterion.cuda()

	# initial best loss
	best_val_loss = np.inf

	# initialize visdom graph
	#vis_train = Visdom()
	#vis_valid = Visdom()

	#train_loss_list = []
	#valid_loss_list = []

	if args.gpu == 0:
		randidx = '{}'.format(np.random.randint(0, 10000)).zfill(4)
		model_name = 'transformer-s{}-t{}-b{}-n{}-md{}-ff{}-h{}-r{}.bin'.format(
			args.inp_n_words,		# s  : source vocab count
			args.out_n_words,		# t  : target vocab count
			args.train_batch_size,	# b  : batch size
			args.N,					# n  : number of layers
			args.d_model,			# md : d_model
			args.d_ff,				# ff : d_ff
			args.h,					# h  : hidden size
			randidx)				# r  : random number
	else:
		model_name = 'a.bin'
	print_log('model name to be saved: {}'.format(os.path.join(args.model_path, model_name)))

	for epoch in range(args.epochs):
		train_losses = train_epoch(
			(rebatch(pad_id, b) for b in train_iter), 
			model, 
			criterion, 
			optimizer, 
			args.gpu,
			epoch,
			args.fp16)
		valid_loss = valid_epoch(
			(rebatch(pad_id, b) for b in valid_iter), 
			model, 
			criterion, 
			optimizer, 
			epoch,
			args.fp16)

		sum_of_weight = sum([p[1].data.sum() for p in model.named_parameters()])
		print_log('GPU{} -> sum_of_weight={:.4f}'.format(args.gpu, sum_of_weight))

		if args.gpu == 0:
			if valid_loss >= best_val_loss:
				print_log('Try again. Current best is still {:.4f} (< {:.4f})'.format(best_val_loss, valid_loss))
			else:
				print_log('New record. from {:.4f} to {:.4f}'.format(best_val_loss, valid_loss))
				best_val_loss = valid_loss
				save_model(args, model, optimizer, epoch, valid_loss, model_name=model_name)

		# blocking processes
		torch.distributed.barrier()

if __name__ == '__main__':
	main()


