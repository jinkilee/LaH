#!/usr/bin/env python
# coding: utf-8

# Reference
# - https://nlp.seas.harvard.edu/2018/04/03/attention.html

# In[1]:


import sys
sys.path.insert(0, '../')

import os
from tqdm import tqdm, trange
import glob
import setting
import numpy as np
import sentencepiece as spm
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
import logging
import logging.config
from utils import fix_torch_randomness
from modeling.transformer import *
from dataset import load_dataset
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel as DDP
from conf import *
from feature import *
from optimizer import NoamOpt

# define logger 
logging.config.fileConfig('logging.conf')
log = logging.getLogger('LaH')

fileHandler = logging.FileHandler('train.log')
log.addHandler(fileHandler)

# define ENV variables
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '10002'

def make_model(src_vocab, tgt_vocab, N=6, 
			   d_model=512, d_ff=2048, h=8, dropout=0.1):
	"Helper: Construct a model from hyperparameters."
	c = copy.deepcopy
	attn = MultiHeadedAttention(h, d_model)
	ff = PositionwiseFeedForward(d_model, d_ff, dropout)
	position = PositionalEncoding(d_model, dropout)
	model = EncoderDecoder(
		Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
		Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
		nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
		nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
		Generator(d_model, tgt_vocab))
	
	# This was important from their code. 
	# Initialize parameters with Glorot / fan_avg.
	for p in model.parameters():
		if p.dim() > 1:
			nn.init.xavier_uniform(p)
	return model

make_sentencepiece = True
src_cmd = templates.format(src_input_file,
				pad_id,
				bos_id,
				eos_id,
				unk_id,
				src_prefix,
				vocab_size,
				character_coverage,
				model_type)

trg_cmd = templates.format(trg_input_file,
				pad_id,
				bos_id,
				eos_id,
				unk_id,
				trg_prefix,
				vocab_size,
				character_coverage,
				model_type)

def get_sentencepiece(src_prefix, trg_prefix, src_cmd=None, trg_cmd=None):
	if make_sentencepiece:
		src_spm = spm.SentencePieceTrainer.Train(src_cmd)
		trg_spm = spm.SentencePieceTrainer.Train(trg_cmd)
		src_spm = spm.SentencePieceProcessor()
		trg_spm = spm.SentencePieceProcessor()
		src_spm.Load('{}.model'.format(src_prefix)) 
		trg_spm.Load('{}.model'.format(trg_prefix))
	else: 
		src_spm = spm.SentencePieceProcessor()
		trg_spm = spm.SentencePieceProcessor()
		src_spm.Load('{}.model'.format(src_prefix)) 
		trg_spm.Load('{}.model'.format(trg_prefix)) 

	extra_options = 'bos:eos' #'reverse:bos:eos'
	src_spm.SetEncodeExtraOptions(extra_options)
	trg_spm.SetEncodeExtraOptions(extra_options)

	return src_spm, trg_spm

def set_padding(dataset_list):
	# to list
	src_token_list = [ds[0] for ds in dataset_list]
	trg_token_list = [ds[1] for ds in dataset_list]
	src_mask_list = [ds[2] for ds in dataset_list]
	trg_mask_list = [ds[3] for ds in dataset_list]
	ntokens = sum([ds[4] for ds in dataset_list])
	
	# padding
	src_token_tensor = pad_sequence(src_token_list, batch_first=True)
	trg_token_tensor = pad_sequence(trg_token_list, batch_first=True)
	src_mask_tensor = pad_sequence(src_mask_list, batch_first=True).unsqueeze(dim=-2)
	trg_mask_tensor = pad_sequence(trg_mask_list, batch_first=True).unsqueeze(dim=-2)

	return [src_token_tensor, trg_token_tensor, src_mask_tensor, trg_mask_tensor, ntokens]

class TranslationDataset(Dataset):
	def __init__(self, sent_pairs, src_spm, trg_spm):
		self.dataset = sent_pairs
		self.src_spm = src_spm
		self.trg_spm = trg_spm

	def __len__(self):
		return len(self.dataset)
	
	def __getitem__(self, idx):
		src, trg = self.dataset[idx]
		
		# to Tensor
		inp_tensor = torch.LongTensor(self.src_spm.EncodeAsIds(src))#.unsqueeze(dim=0)
		out_tensor = torch.LongTensor(self.trg_spm.EncodeAsIds(trg))#.unsqueeze(dim=0)
		src_mask = (inp_tensor != self.src_spm.pad_id()).int()
		trg_mask = (out_tensor != self.trg_spm.pad_id()).int()
		ntokens = (out_tensor != self.trg_spm.pad_id()).data.sum()
		
		# to Variable
		src_token = Variable(inp_tensor, requires_grad=False)
		trg_token = Variable(out_tensor, requires_grad=False)
		src_mask = Variable(src_mask, requires_grad=False)
		trg_mask = Variable(trg_mask, requires_grad=False)
		
		return [src_token, trg_token, src_mask, trg_mask, ntokens]

def to_gpu(batch, device):
	return list(map(lambda b: b.to(device), batch))

def do_train(dataloader, model, loss_compute, device, epoch):
	# change model to train mode
	model.train()

	start = time.time()
	total_tokens = 0
	total_loss = 0
	tokens = 0

	with tqdm(dataloader, desc='training {}th epoch'.format(epoch)) as tbar:
		for i, batch in enumerate(tbar):
			# run model for training
			if device == 'cpu':
				batch_src, batch_trg, batch_src_mask, batch_trg_mask, batch_ntokens = batch
			else:
				batch_src, batch_trg, batch_src_mask, batch_trg_mask, batch_ntokens = to_gpu(batch, device)
			out = model.forward(batch_src, batch_trg, batch_src_mask, batch_trg_mask)

			# calculate loss
			loss = loss_compute(out, batch_trg, batch_ntokens)
			total_loss += loss
			total_tokens += batch_ntokens
			tokens += batch_ntokens

			# print loss
			if i % 50 == 1:
				elapsed = time.time() - start
				start = time.time()
				tokens = 0
		
			# update tbar
			tbar.set_postfix(loss=(total_loss/total_tokens).data.item())

def do_valid(dataloader, model, loss_compute, device, epoch):
	# change model to validation mode
	model.eval()

	total_tokens = 0
	total_loss = 0
	tokens = 0
	with tqdm(dataloader, desc='validating {}th epoch'.format(epoch)) as tbar:
		for i, batch in enumerate(tbar):
			# run model for training
			if device == 'cpu':
				batch_src, batch_trg, batch_src_mask, batch_trg_mask, batch_ntokens = batch
			else:
				batch_src, batch_trg, batch_src_mask, batch_trg_mask, batch_ntokens = to_gpu(batch, device)
			out = model.forward(batch_src, batch_trg, batch_src_mask, batch_trg_mask)

			# calculate loss
			loss = loss_compute(out, batch_trg, batch_ntokens, do_backward=False)
			total_loss += loss
			total_tokens += batch_ntokens
			tokens += batch_ntokens

			# update tbar
			tbar.set_postfix(loss=(total_loss/total_tokens).data.item())

		return total_loss / total_tokens

def do_save(model, optimizer, epoch, loss):
	model_full_path = './models/model-tmp.bin'

	# FIXME: fix model name
	torch.save({
		'epoch': epoch + 1,					  # need only for retraining
		'state_dict': model.module.state_dict(),
		'best_val_loss': loss,		  # need only for retraining
		'optimizer' : optimizer.optimizer.state_dict(), # need only for retraining
		'learning_rate' : optimizer._rate, # need only for retraining
	}, model_full_path)

	sum_of_weight = sum([p[1].data.sum() for p in model.named_parameters()])
	log.info('model was saved at {} -> {:.4f}'.format(model_full_path, sum_of_weight))

class LabelSmoothing(nn.Module):
	"Implement label smoothing."
	def __init__(self, size, padding_idx, smoothing=0.0):
		super(LabelSmoothing, self).__init__()
		self.criterion = nn.KLDivLoss(size_average=False)
		self.padding_idx = padding_idx
		self.confidence = 1.0 - smoothing
		self.smoothing = smoothing
		self.size = size
		self.true_dist = None
		
	def forward(self, x, target):
		assert x.size(1) == self.size
		true_dist = x.data.clone()
		true_dist.fill_(self.smoothing / (self.size - 2))
		true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
		true_dist[:, self.padding_idx] = 0
		mask = torch.nonzero(target.data == self.padding_idx)
		if mask.dim() > 0:
			true_dist.index_fill_(0, mask.squeeze(), 0.0)
		self.true_dist = true_dist
		return self.criterion(x, Variable(true_dist, requires_grad=False))


class SimpleLossCompute:
	"A simple loss compute and train function."
	def __init__(self, generator, criterion, opt=None):
		self.generator = generator
		self.criterion = criterion
		self.opt = opt
		
	def __call__(self, x, y, norm, do_backward=True):
		x = self.generator(x)
		a = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1))
		loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
							  y.contiguous().view(-1)) / norm
		if not do_backward:
			return loss.data.item() * norm.float()

		if self.opt is not None:
			self.opt.step()
			self.opt.optimizer.zero_grad()
		loss.backward()
		return loss.data.item() * norm.float()

parser = argparse.ArgumentParser()
parser.add_argument('--world_size', default=setting.world_size, type=int)
parser.add_argument('--multi_gpu', default=setting.multi_gpu, type=bool)
parser.add_argument('--n_words', default=setting.n_words, type=int)

def main():
	args = parser.parse_args()

	if args.multi_gpu:
		ngpus_per_node = torch.cuda.device_count()
	else:
		ngpus_per_node = 1

	args.world_size = ngpus_per_node
	mp.spawn(main_worker, nprocs=ngpus_per_node, 
			 args=(ngpus_per_node, args, ))

def main_worker(gpu, ngpus_per_node, args):
	global best_acc1
	args.gpu = gpu

	torch.cuda.set_device(args.gpu)
	dist.init_process_group(backend='nccl', 
							init_method='env://',
							world_size=args.world_size,
							rank=args.gpu)

	# load dataset
	sent_pairs = load_dataset(path='/heavy_data/jkfirst/workspace/git/LaH/dataset/')
	sent_pairs = list(map(lambda x: remove_bos_eos(x), sent_pairs))

	# split train/valid sentence pairs
	n_train = int(len(sent_pairs) * 0.8)
	np.random.seed(args.gpu)
	train_sent_pairs = sent_pairs[:n_train]
	np.random.shuffle(train_sent_pairs)
	train_sent_pairs = train_sent_pairs[:int(n_train*(1.-1./args.world_size))]
	valid_sent_pairs = sent_pairs[n_train:]

	# make dataloader with dataset
	# FIXME: RuntimeError: Internal: unk is not defined.
	src_spm, trg_spm = get_sentencepiece(src_prefix, trg_prefix, src_cmd=src_cmd, trg_cmd=trg_cmd)
	train_dataset = TranslationDataset(train_sent_pairs, src_spm, trg_spm)
	valid_dataset = TranslationDataset(valid_sent_pairs, src_spm, trg_spm)
	train_dataloader = DataLoader(train_dataset, batch_size=128, collate_fn=set_padding)
	valid_dataloader = DataLoader(valid_dataset, batch_size=100, collate_fn=set_padding)

	# fix torch randomness
	fix_torch_randomness()

	# Train the simple copy task.
	criterion = LabelSmoothing(size=args.n_words, padding_idx=0, smoothing=0.0)
	model = make_model(args.n_words, args.n_words)
	optimizer = NoamOpt(model.src_embed[0].d_model, 1, 400,
			torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

	# make gpu-distributed model
	device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
	model.to(device)
	model = DDP(model, device_ids=[args.gpu])

	best_val_loss = np.inf
	for epoch in range(10):
		# FIXME: add sampler
		do_train(train_dataloader,
				model, 
				SimpleLossCompute(model.module.generator, criterion, optimizer),
				device,
				epoch)
		valid_loss = do_valid(valid_dataloader,
				model,
				SimpleLossCompute(model.module.generator, criterion, optimizer),
				device,
				epoch)

		if valid_loss >= best_val_loss:
			log.info('Try again. Current best is still {:.4f}'.format(best_val_loss))
		else:
			log.info('New record. from {:.4f} to {:.4f}'.format(best_val_loss, valid_loss))
			best_val_loss = valid_loss
			if args.gpu == 0:
				do_save(model, optimizer, epoch, best_val_loss)

if __name__ == '__main__':
	main()






