#!/usr/bin/env python
# coding: utf-8

# Reference
# - https://nlp.seas.harvard.edu/2018/04/03/attention.html

# In[1]:


import sys
sys.path.insert(0, '../modeling')

import os
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

from torchtext import data, datasets
from loss_compute import SimpleLossCompute, MultiGPULossCompute
from utils import fix_torch_randomness, to_gpu, get_sentencepiece
from transformer import *
from dataset import load_dataset, set_padding, KRENDataset, Batch
from torch.utils.data import DataLoader
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
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '10002'

inp_lang, out_lang = get_sentencepiece(src_prefix, trg_prefix)

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

make_sentencepiece = True
src_cmd = templates.format(src_input_file,
				pad_id,
				bos_id,
				eos_id,
				unk_id,
				src_prefix,
				src_vocab_size,
				character_coverage,
				model_type)

trg_cmd = templates.format(trg_input_file,
				pad_id,
				bos_id,
				eos_id,
				unk_id,
				trg_prefix,
				trg_vocab_size,
				character_coverage,
				model_type)

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
			log.debug('{} {}'.format(out.shape, batch.trg_y.shape))
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
			batch_src, batch_trg, batch_src_mask, batch_trg_mask, batch_ntokens = batch
			out = model.forward(batch_src, batch_trg, batch_src_mask, batch_trg_mask)

			# calculate loss
			loss = loss_compute(out, batch_trg, batch_ntokens)
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


parser = argparse.ArgumentParser()

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

def tokenize_kr(text):
	return inp_lang.EncodeAsPieces(text)

def tokenize_en(text):
	return out_lang.EncodeAsPieces(text)

def main():
	args = parser.parse_args()

	args.gpu = 0
	#devices = range(torch.cuda.device_count())
	devices = [0, 1]

	# load dataset
	sent_pairs = load_dataset(path='/heavy_data/jkfirst/workspace/git/LaH/dataset/')

	# split train/valid sentence pairs
	n_train = int(len(sent_pairs) * 0.8)
	train_sent_pairs = sent_pairs[:n_train]
	valid_sent_pairs = sent_pairs[n_train:]
	log.debug('{} {}'.format(len(train_sent_pairs), len(valid_sent_pairs)))
	train_sent_pairs = sorted(train_sent_pairs, key=lambda x: (len(x[0]), len(x[1])))

	SRC = data.Field(tokenize=tokenize_kr, pad_token='<pad>')
	TRG = data.Field(tokenize=tokenize_en, init_token='<bos>', eos_token='<eos>', pad_token='<pad>')

	# make dataloader with dataset
	train, valid, test = KRENDataset.splits(sent_pairs, (SRC, TRG), inp_lang, out_lang)

	SRC.build_vocab(train.src, min_freq=MIN_FREQ)
	TRG.build_vocab(train.trg, min_freq=MIN_FREQ)

	train_iter = MyIterator(train, batch_size=128, device=0,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
	valid_iter = MyIterator(valid, batch_size=100, device=0,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)

	# fix torch randomness
	fix_torch_randomness()

	# Train the simple copy task.
	args.inp_n_words = src_vocab_size
	args.out_n_words = trg_vocab_size
	log.info('inp_n_words: {} out_n_words: {}'.format(args.inp_n_words, args.out_n_words))
	model = make_model(args.inp_n_words, args.out_n_words)
	model.cuda()

	criterion = LabelSmoothing(size=args.out_n_words, padding_idx=0, smoothing=0.0)
	criterion.cuda()

	optimizer = NoamOpt(
			model.src_embed[0].d_model, 
			1, 
			400,
			torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

	# make parallel model
	model_par = nn.DataParallel(model, device_ids=[0,1,2,3])
	#model_par = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

	best_val_loss = np.inf
	for epoch in range(45):
		# FIXME: add sampler
		do_train((rebatch(pad_id, b) for b in train_iter),
				model_par, 
				MultiGPULossCompute(model.generator, criterion, devices=devices, opt=optimizer),
				epoch)
		valid_loss = do_valid(valid_dataloader,
				model_par,
				MultiGPULossCompute(model.generator, criterion, devices=devices, opt=None),
				epoch)

		if valid_loss >= best_val_loss:
			log.info('Try again. valid_loss is not smaller than current best: {:.6f} > {:.6f}'.format(valid_loss, best_val_loss))
		else:
			log.info('New record. from {:.6f} to {:.6f}'.format(best_val_loss, valid_loss))
			best_val_loss = valid_loss
			if args.gpu == 0:
				do_save(model, optimizer, epoch, best_val_loss)

if __name__ == '__main__':
	main()

#from loss_compute import SimpleLossCompute, MultiGPULossCompute
