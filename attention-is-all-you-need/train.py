#!/usr/bin/env python
# coding: utf-8

# Reference
# - https://nlp.seas.harvard.edu/2018/04/03/attention.html

import sys
sys.path.insert(0, '../modeling')

import os
import dill
import time
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
import random

from torchtext import data, datasets
from loss_compute import SimpleLossCompute, MultiGPULossCompute, MultiGPULossCompute_
from utils import fix_torch_randomness, to_gpu, get_sentencepiece
from transformer import *
from dataset import load_dataset_aihub, set_padding, KRENDataset, KRENField, Batch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from setting import *
from feature import *
from optimizer import NoamOpt
from torch.autograd import Variable

# define logger 
logging.config.fileConfig('logging.conf')
log = logging.getLogger('LaH')

fileHandler = logging.FileHandler('train.log')
log.addHandler(fileHandler)

# define ENV variables
#os.environ['MASTER_ADDR'] = '127.0.0.1'
#os.environ['MASTER_PORT'] = '10002'


parser = argparse.ArgumentParser()
parser.add_argument('--devices', default='0', type=str)
parser.add_argument('--epochs', default=45, type=int)

def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

def rebatch(pad_idx, batch):
	"Fix order in torchtext to match ours"
	src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
	return Batch(src, trg, pad_idx)

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
			#nn.init.xavier_uniform(p)
			nn.init.xavier_uniform_(p)
	return model

def do_train(dataloader, model, loss_compute, epoch):
	# change model to train mode
	model.train()

	start = time.time()
	total_tokens = 0
	total_loss = 0
	tokens = 0

	with tqdm(dataloader, desc='training {}th epoch'.format(epoch)) as tbar:
		for i, batch in enumerate(tbar):
			#batch_src, batch_trg, batch_src_mask, batch_trg_mask, batch_ntokens = batch
			out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)

			# calculate loss
			loss = loss_compute(out, batch.trg_y, batch.ntokens)
			total_loss += loss
			total_tokens += batch.ntokens
		
			# update tbar
			tbar.set_postfix(loss=(total_loss/total_tokens).data.item())

def do_valid(dataloader, model, loss_compute, epoch):
	# change model to validation mode
	model.eval()

	total_tokens = 0
	total_loss = 0
	tokens = 0
	with tqdm(dataloader, desc='validating {}th epoch'.format(epoch)) as tbar:
		for i, batch in enumerate(tbar):
			#batch_src, batch_trg, batch_src_mask, batch_trg_mask, batch_ntokens = batch
			out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)

			# calculate loss
			loss = loss_compute(out, batch.trg_y, batch.ntokens)
			total_loss += loss
			total_tokens += batch.ntokens
			tokens += batch.ntokens

			# update tbar
			tbar.set_postfix(loss=(total_loss/total_tokens).data.item())

		return total_loss / total_tokens

def do_save(model, optimizer, epoch, loss):
	model_full_path = './models/model-tmp.bin'

	# FIXME: fix model name
	torch.save({
		'epoch': epoch + 1,					  # need only for retraining
		'state_dict': model.state_dict(),
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


class MyIterator(data.Iterator):
	def create_batches(self):
		if self.train:
			def pool(d, random_shuffler):
				for p in data.batch(d, self.batch_size * 100):
					p_batch = data.batch(
						sorted(p, key=self.sort_key),
						self.batch_size, self.batch_size_fn)
					for b in random_shuffler(list(p_batch)):
						yield b
			self.batches = pool(self.data(), self.random_shuffler)

		else:
			self.batches = []
			for b in data.batch(self.data(), self.batch_size, self.batch_size_fn):
				self.batches.append(sorted(b, key=self.sort_key))

def main():
	args = parser.parse_args()
	if args.devices is not 'cpu':
		devices = list(map(int, args.devices.split(',')))

	# load dataset
	sent_pairs = load_dataset_aihub()
	#sent_pairs = load_dataset_aihub(path='/heavy_data/jkfirst/workspace/git/LaH/attention-is-all-you-need/data')
	inp_lang, out_lang = get_sentencepiece(src_prefix, trg_prefix)
	log.info('loaded input sentencepiece model: {}'.format(src_prefix))
	log.info('loaded output sentencepiece model: {}'.format(trg_prefix))

	# shuffle sent_pairs
	random.seed(100)
	random.shuffle(sent_pairs)

	# split train/valid sentence pairs
	n_train = int(len(sent_pairs) * 0.8)
	train_sent_pairs = sent_pairs[:n_train]
	valid_sent_pairs = sent_pairs[n_train:]
	train_sent_pairs = sorted(train_sent_pairs, key=lambda x: (len(x[0]), len(x[1])))

	# these are used for defining tokenize method and some reserved words
	SRC = KRENField(
		tokenize=inp_lang.EncodeAsPieces, 
		pad_token='<pad>')
	TRG = KRENField(
		tokenize=out_lang.EncodeAsPieces, 
		init_token='<s>', 
		eos_token='</s>', 
		pad_token='<pad>')

	# make dataloader from KRENDataset
	train, valid, test = KRENDataset.splits(sent_pairs, (SRC, TRG), inp_lang, out_lang)
	SRC.build_vocab(train.src)
	TRG.build_vocab(train.trg)
	log.debug(SRC.vocab)

	torch.save(SRC.vocab, 'spm/{}.spm'.format(src_prefix), pickle_module=dill)
	torch.save(TRG.vocab, 'spm/{}.spm'.format(trg_prefix), pickle_module=dill)
	log.info('input vocab was saved: spm/{}.spm'.format(src_prefix))
	log.info('output vocab was saved: spm/{}.spm'.format(trg_prefix))

	train_iter = MyIterator(train, batch_size=512, device=0,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
	valid_iter = MyIterator(valid, batch_size=100, device=0,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)

	# fix torch randomness
	fix_torch_randomness()

	# define input/output size
	args.inp_n_words = src_vocab_size
	args.out_n_words = trg_vocab_size
	log.info('inp_n_words: {} out_n_words: {}'.format(args.inp_n_words, args.out_n_words))

	# define model
	model = make_model(args.inp_n_words, args.out_n_words)
	if args.devices is not 'cpu':
		model.cuda()

	# define model
	criterion = LabelSmoothing(size=args.out_n_words, padding_idx=0, smoothing=0.0)
	if args.devices is not 'cpu':
		criterion.cuda()

	# define optimizer
	optimizer = NoamOpt(
			model_size=model.src_embed[0].d_model, 
			factor=1, 
			warmup=400,
			optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

	# make parallel model
	if args.devices is not 'cpu' and len(devices) > 1:
		model_par = nn.DataParallel(model, device_ids=devices)

	# initial best loss
	best_val_loss = np.inf

	# start training
	for epoch in range(args.epochs):
		#'''
		# FIXME: add sampler
		do_train((rebatch(pad_id, b) for b in train_iter),
				model_par if args.devices is not 'cpu' and len(devices) > 1 else model,
				MultiGPULossCompute_(model.generator, criterion, devices=devices, opt=optimizer),
				epoch)
		#'''
		valid_loss = do_valid((rebatch(pad_id, b) for b in valid_iter),
				model_par if args.devices is not 'cpu' and len(devices) > 1 else model,
				MultiGPULossCompute_(model.generator, criterion, devices=devices, opt=None),
				epoch)

		if valid_loss >= best_val_loss:
			log.info('Try again. valid_loss is not smaller \
				than current best: {:.6f} > {:.6f}'.format(valid_loss, best_val_loss))
		else:
			log.info('New record. from {:.6f} to {:.6f}'.format(best_val_loss, valid_loss))
			best_val_loss = valid_loss
			do_save(model, optimizer, epoch, best_val_loss)

if __name__ == '__main__':
	main()

#from loss_compute import SimpleLossCompute, MultiGPULossCompute
