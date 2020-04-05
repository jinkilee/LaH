import sys
sys.path.insert(0, '../modeling')

import os
import dill
import random
import glob
import setting
import torch
import numpy as np
import sentencepiece as spm
import argparse
import torch.distributed as dist
#import torch.multiprocessing as mp
import logging
import logging.config

from visdom import Visdom
from utils import fix_torch_randomness, get_sentencepiece 
from transformer import make_model, subsequent_mask
from dataset import load_dataset_aihub, KRENDataset, KRENField, MyIterator, batch_size_fn 
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.autograd import Variable
from setting import *
from feature import *

#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# define logger 
logging.config.fileConfig('logging.conf')
log = logging.getLogger('LaH')

fileHandler = logging.FileHandler('evaluate.log')
log.addHandler(fileHandler)

# define ENV variables
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '10002'


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
		next_word = next_word.item()
		ys = torch.cat([ys, 
						torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
	return ys

parser = argparse.ArgumentParser()
parser.add_argument('--world_size', default=setting.world_size, type=int)
parser.add_argument('--multi_gpu', default=setting.multi_gpu, type=bool)
parser.add_argument('--modelnm', default='a.bin', type=str)
parser.add_argument('--N', default=setting.N, type=int)
parser.add_argument('--d_model', default=setting.d_model, type=int)
parser.add_argument('--d_ff', default=setting.d_ff, type=int)
parser.add_argument('--h', default=setting.h, type=int)
parser.add_argument('--dropout', default=0.0, type=float)
parser.add_argument('--small_model', default=False, type=bool)

def main():
	args = parser.parse_args()

	# load dataset
	#sent_pairs = load_dataset_aihub()
	sent_pairs = load_dataset_aihub(path='data/')
	#random.seed(100)
	#random.shuffle(sent_pairs)

	# make dataloader with dataset
	# FIXME: RuntimeError: Internal: unk is not defined.
	inp_lang, out_lang = get_sentencepiece(src_prefix, trg_prefix)
	log.info('loaded input sentencepiece model: {}'.format(src_prefix))
	log.info('loaded output sentencepiece model: {}'.format(trg_prefix))

	# split train/valid sentence pairs
	n_train = int(len(sent_pairs) * 0.8)
	valid_sent_pairs = sent_pairs[n_train:]
	log.info('valid_sent_pairs: {}'.format(len(valid_sent_pairs)))

	# these are used for defining tokenize method and some reserved words
	SRC = KRENField(pad_token='<pad>')
	TRG = KRENField(pad_token='<pad>')

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

	# make dataloader from KRENDataset
	train, valid, test = KRENDataset.splits(sent_pairs, (SRC, TRG), inp_lang, out_lang, encoding_type='pieces')
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
	if args.small_model:
		model = make_model(
			args.inp_n_words, 
			args.out_n_words,
			dropout=args.dropout)
	else:
		model = make_model(
			args.inp_n_words, 
			args.out_n_words,
			N=N,
			d_model=args.d_model,
			d_ff=args.d_ff,
			h=args.h,
			dropout=args.dropout)

	#model_name_full_path = './models/model-tmp.bin'
	model_name_full_path = args.modelnm
	checkpoint = torch.load(model_name_full_path)
	state_dict = checkpoint['state_dict']
	model.load_state_dict(state_dict)
	model.cuda()
	
	model.eval()
	for i, batch in enumerate(valid_iter):
		src = batch.src.transpose(0, 1)[:1]
		src_mask = (src != SRC.vocab.stoi["<pad>"]).unsqueeze(-2)
		print("Input::", end="\t")
		for i in range(src.size(1)):
			sym = SRC.vocab.itos[src[0, i]]
			if sym == "</s>": break
			print(sym, end =" ")
		print('')
		
		out = greedy_decode(model, src.cuda(), src_mask.cuda(), 
							max_len=60, start_symbol=TRG.vocab.stoi["<s>"])
		print("Translation:", end="\t")
		for i in range(1, out.size(1)):
			sym = TRG.vocab.itos[out[0, i]]
			if sym == "</s>": break
			print(sym, end =" ")
		print('')
		print("Target:", end="\t")
		for i in range(1, batch.trg.size(0)):
			sym = TRG.vocab.itos[batch.trg.data[i, 0]]
			if sym == "</s>": break
			print(sym, end =" ")
		print('')
		print('---------------')

if __name__ == '__main__':
	main()


