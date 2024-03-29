import sentencepiece as spm
import torch
from functools import reduce

def fix_torch_randomness(seed=0):
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

def to_gpu(batch):
	return list(map(lambda b: b.cuda(), batch))
	#return list(map(lambda b: b.to(device), batch))

def get_sentencepiece(src_prefix, trg_prefix, src_cmd=None, trg_cmd=None, make_sentencepiece=False):
	if make_sentencepiece:
		src_spm = spm.SentencePieceTrainer.Train(src_cmd)
		trg_spm = spm.SentencePieceTrainer.Train(trg_cmd)
		src_spm = spm.SentencePieceProcessor()
		trg_spm = spm.SentencePieceProcessor()
		src_spm.Load('spm/{}.model'.format(src_prefix)) 
		trg_spm.Load('spm/{}.model'.format(trg_prefix))
	else: 
		src_spm = spm.SentencePieceProcessor()
		trg_spm = spm.SentencePieceProcessor()
		src_spm.Load('spm/{}.model'.format(src_prefix)) 
		trg_spm.Load('spm/{}.model'.format(trg_prefix)) 

	extra_options = 'bos:eos' #'reverse:bos:eos'
	#src_spm.SetEncodeExtraOptions(extra_options)
	trg_spm.SetEncodeExtraOptions(extra_options)

	return src_spm, trg_spm

def get_number_of_params(model):
	n_params = 0
	for parameter in model.parameters():
		n_params += reduce(lambda x, y: x*y, parameter.shape)
	return n_params
