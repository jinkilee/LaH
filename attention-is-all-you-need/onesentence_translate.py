import sys
sys.path.insert(0, '../modeling')

import os
import dill
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

from visdom import Visdom
from tqdm import tqdm, trange
from loss_compute import SimpleLossCompute, MultiGPULossCompute, MultiGPULossCompute_
from utils import fix_torch_randomness, get_sentencepiece 
from transformer import *
from dataset import load_dataset_aihub, KRENDataset, KRENField, MyIterator, batch_size_fn, rebatch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from setting import *
from feature import *
from optimizer import NoamOpt

# define logger 
logging.config.fileConfig('logging.conf')
log = logging.getLogger('LaH')

fileHandler = logging.FileHandler('retrain.log')
log.addHandler(fileHandler)

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

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

def do_translate(valid_iter, model):
	for i, batch in enumerate(valid_iter):
		if i == 10:
			break
		src = batch.src.transpose(0, 1)[:1]
		trg = batch.trg.transpose(0, 1)[:1]
		print(src.shape)
		print(src)
		print(trg)
		print('--------')

	'''
		exit()
		src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
		out = greedy_decode(model, src, src_mask, 
							max_len=60, start_symbol=TGT.vocab.stoi['<s>'])
		print("Translation:", end="\t")
		for i in range(1, out.size(1)):
			sym = TGT.vocab.itos[out[0, i]]
			if sym == "</s>": break
			print(sym, end =" ")
		print()
		print("Target:", end="\t")
		for i in range(1, batch.trg.size(0)):
			sym = TGT.vocab.itos[batch.trg.data[i, 0]]
			if sym == "</s>": break
			print(sym, end =" ")
		print()
		break
	'''

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
parser.add_argument('--epochs', default=45, type=int)
#parser.add_argument('--n_words', default=setting.n_words, type=int)

# define tokenizer
def src_tokenize(x):
	tokens = [SRC.vocab.stoi['<unk>']] * len(x)
	for i, xi in enumerate(x):
		try:
			tokens[i] = xi
		except KeyError:
			pass
	return tokens

def trg_tokenize(x):
	tokens = [TRG.vocab.stoi['<unk>']] * len(x)
	print(tokens)
	for i, xi in enumerate(x):
		try:
			tokens[i] = xi
		except KeyError:
			pass
	return tokens

def main():
	args = parser.parse_args()

	# load dataset
	#sent_pairs = load_dataset_aihub()
	sent_pairs = load_dataset_aihub(path='data/')
	random.seed(100)
	random.shuffle(sent_pairs)

	# make dataloader with dataset
	# FIXME: RuntimeError: Internal: unk is not defined.
	inp_lang, out_lang = get_sentencepiece(src_prefix, trg_prefix)
	log.info('loaded input sentencepiece model: {}'.format(src_prefix))
	log.info('loaded output sentencepiece model: {}'.format(trg_prefix))

	# split train/valid sentence pairs
	n_train = int(len(sent_pairs) * 0.8)
	valid_sent_pairs = sent_pairs[n_train:]
	#log.info('train_sent_pairs: {}'.format(len(train_sent_pairs)))
	log.info('valid_sent_pairs: {}'.format(len(valid_sent_pairs)))

	# these are used for defining tokenize method and some reserved words
	SRC = KRENField(
		pad_token='<pad>')
	TRG = KRENField(
		pad_token='<pad>')

	# load SRC/TRG
	if not os.path.exists('spm/{}.model'.format(src_prefix)) or \
		not os.path.exists('spm/{}.model'.format(trg_prefix)):
		# build vocabulary
		SRC.build_vocab(train.src)
		TRG.build_vocab(train.trg)
		torch.save(SRC.vocab, 'spm/{}.spm'.format(src_prefix), pickle_module=dill)
		torch.save(TRG.vocab, 'spm/{}.spm'.format(trg_prefix), pickle_module=dill)
		log.info('input vocab was created and saved: spm/{}.spm'.format(src_prefix))
		log.info('output vocab was created and saved: spm/{}.spm'.format(trg_prefix))
	else:
		src_vocab = torch.load('spm/{}.spm'.format(src_prefix), pickle_module=dill)
		trg_vocab = torch.load('spm/{}.spm'.format(trg_prefix), pickle_module=dill)
		SRC.vocab = src_vocab
		TRG.vocab = trg_vocab
		log.info('input vocab was loaded: spm/{}.spm'.format(src_prefix))
		log.info('output vocab was loaded: spm/{}.spm'.format(trg_prefix))

	# define tokenizer
	SRC.tokenize = src_tokenize
	TRG.tokenize = trg_tokenize

	# make dataloader from KRENDataset
	train, valid, test = KRENDataset.splits(sent_pairs, (SRC, TRG), inp_lang, out_lang)
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

	# load model
	model_name_full_path = './models/model-tmp.bin'
	checkpoint = torch.load(model_name_full_path)
	state_dict = checkpoint['state_dict']
	model.load_state_dict(state_dict)
	model.cuda()

	sum_of_weight = sum([p[1].data.sum() for p in model.named_parameters()])
	log.info('model was successfully loaded: {:.4f}'.format(sum_of_weight))

	#(rebatch(pad_id, b) for b in valid_iter),
	do_translate(valid_iter, model)

if __name__ == '__main__':
	main()

