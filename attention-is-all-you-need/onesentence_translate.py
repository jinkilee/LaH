import sys
sys.path.insert(0, '../modeling')

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

from torchtext import data, datasets
from torchtext.data.metrics import bleu_score
from utils import fix_torch_randomness, get_sentencepiece, to_gpu
from transformer import *
from dataset import load_dataset, set_padding, KRENDataset, KRENField, Batch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel as DDP
from setting import *
from feature import *
from optimizer import NoamOpt

# define logger 
logging.config.fileConfig('logging.conf')
log = logging.getLogger('LaH')

fileHandler = logging.FileHandler('onesentence_translate.log')
log.addHandler(fileHandler)

# define ENV variables
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '10002'

parser = argparse.ArgumentParser()
parser.add_argument('--devices', default='0', type=str)

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
			nn.init.xavier_uniform(p)
	return model

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

def do_translate(dataloader, model, epoch):
	# change model to validation mode
	model.eval()

	original_input = []
	translated_ids = []
	translated_lbl = []
	with tqdm(dataloader, desc='validating {}th epoch'.format(epoch)) as tbar:
		for i, batch in enumerate(tbar):
			if i == 10:
				break
			log.debug(batch)
			# run model for training
			out = greedy_decode(
				model.module, 
				batch.src.cuda(), 
				batch.src_mask.cuda(), 
				max_len=60, 
				start_symbol=bos_id)

			log.debug('x: {}'.format(batch.src))
			log.debug('p: {}'.format(out))
			log.debug('y: {}'.format(batch.trg))
			log.debug('-------------')

			original_input.extend(batch.src.to('cpu').numpy().tolist())
			translated_ids.extend(out.to('cpu').numpy().tolist())
			translated_lbl.extend(batch.trg.to('cpu').numpy().tolist())
			#log.debug('out: {}'.format(translated_ids[-1]))
			#log.debug('lbl: {}'.format(translated_lbl[-1]))
	return original_input, translated_ids, translated_lbl

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

class SimpleTranslation:
	"A simple loss compute and train function."
	def __init__(self, generator):
		self.generator = generator
		
	def __call__(self, x):
		x = self.generator(x)
		return x.argmax(dim=-1).int()

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
	else:
		devices = 'cpu'

	# load dataset
	sent_pairs = load_dataset(path='/heavy_data/jkfirst/workspace/git/LaH/dataset/')
	inp_lang, out_lang = get_sentencepiece(src_prefix, trg_prefix)

	# split train/valid sentence pairs
	n_train = int(len(sent_pairs) * 0.8)
	train_sent_pairs = sent_pairs[:n_train]
	valid_sent_pairs = sent_pairs[n_train:]
	log.info('{} {}'.format(len(train_sent_pairs), len(valid_sent_pairs)))
	train_sent_pairs = sorted(train_sent_pairs, key=lambda x: (len(x[0]), len(x[1])))

	# these are used for defining tokenize method and some reserved words
	SRC = KRENField(
		tokenize=inp_lang.EncodeAsPieces, 
		pad_token='<pad>')
	TRG = KRENField(
		tokenize=out_lang.EncodeAsPieces, 
		init_token='<bos>', 
		eos_token='<eos>', 
		pad_token='<pad>')

	# FIXME: didn't found any good way to save Field object. so do the same thing as in train.py
	# FIXME: should find a way to save/load vocab
	# make dataloader from KRENDataset
	MIN_FREQ = 2
	train, valid, test = KRENDataset.splits(
		sent_pairs, 
		(SRC, TRG), 
		inp_lang, 
		out_lang)

	SRC.build_vocab(train.src, min_freq=MIN_FREQ)
	TRG.build_vocab(train.trg, min_freq=MIN_FREQ)

	#src = [oneline.rstrip() for oneline in open('kor_src.txt', 'r')]
	#trg = [oneline.rstrip() for oneline in open('eng_trg.txt', 'r')]
	#sent_pairs = [[s, t] for s, t in zip(src, trg)]

	# make dataloader from KRENDataset
	#test = KRENDataset(sent_pairs, (SRC, TRG), inp_lang, out_lang)
	test_iter = MyIterator(valid, batch_size=10, device=0,
							repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
							batch_size_fn=batch_size_fn, train=False)

	# fix torch randomness
	fix_torch_randomness()

	# FIXME: fix hard-coding
	args.inp_n_words = src_vocab_size
	args.out_n_words = trg_vocab_size
	model = make_model(args.inp_n_words, args.out_n_words, dropout=0.0)
	if args.devices is not 'cpu':
		model.cuda()

	# Train the simple copy task.
	criterion = LabelSmoothing(size=args.out_n_words, padding_idx=0, smoothing=0.0)
	if args.devices is not 'cpu':
		criterion.cuda()

	optimizer = NoamOpt(
			model.src_embed[0].d_model, 
			1, 
			400,
			torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

	# load model
	model_name_full_path = './models/model-tmp.bin'
	device_pairs = zip([0], devices)
	map_location = {'cuda:{}'.format(x): '{}'.format('cuda:{}'.format(y)) for x, y in device_pairs}
	checkpoint = torch.load(model_name_full_path, map_location=map_location)
	state_dict = checkpoint['state_dict']
	model.load_state_dict(state_dict)

	sum_of_weight = sum([p[1].data.sum() for p in model.named_parameters()])
	log.info('model was successfully loaded: {:.4f}'.format(sum_of_weight))

	# make parallel model
	if args.devices is not 'cpu' and len(devices) > 1:
		model_par = nn.DataParallel(model, device_ids=devices)
		log.debug('loaded DataParallel')

	# make gpu-distributed model
	#device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
	#model.to(device)

	'''
	for i, batch in enumerate(test_iter):
		src = batch.src.transpose(0, 1)[:1]
		src_mask = (src != pad_id).unsqueeze(-2)
		out = greedy_decode(model, src.cuda(), src_mask.cuda(), max_len=60, start_symbol=bos_id)

		org = list(map(lambda x: SRC.vocab.itos[x], src[0].numpy()))
		
		log.debug('ORG: {}'.format(org))
		print("Translation:", end="\t")
		for i in range(1, out.size(1)):
			sym = TRG.vocab.itos[out[0, i]]
			if sym == "</s>": break
			print(sym, end =" ")
		print()
		print("Target:", end="\t")
		for i in range(1, batch.trg.size(0)):
			sym = TRG.vocab.itos[batch.trg.data[i, 0]]
			if sym == "</s>": break
			print(sym, end =" ")
		print()
		break
	exit()
	'''
	original_input, translated_ids, translated_lbl = do_translate(
			(rebatch(pad_id, b) for b in test_iter),
			model_par if args.devices is not 'cpu' and len(devices) > 1 else model,
			0)
	log.debug('translated ids: {}'.format(len(translated_ids)))
	log.debug('translated lbl: {}'.format(len(translated_lbl)))
	log.debug('-----------------------')

	log.debug(original_input[0])
	log.debug(translated_ids[0])
	org = list(map(lambda x: SRC.vocab.itos[x], original_input[0]))
	pre = list(map(lambda x: TRG.vocab.itos[x], translated_ids[0]))
	lbl = list(map(lambda x: TRG.vocab.itos[x], translated_lbl[0]))
	log.debug('original: {}'.format(org))
	log.debug('translated: {}'.format(pre))
	log.debug('correct: {}'.format(lbl))
	exit()

	with open('output/model-tmp.out', 'w', encoding='utf-8') as out_f:
		for src, pred, trg in zip(original_input, translated_pred, translated_label):
			src = ''.join(src)
			pred = ''.join(pred)
			trg = ''.join(trg)
			out_f.write('input: {}\n'.format(src))
			out_f.write('pred : {}\n'.format(pred))
			out_f.write('label: {}\n'.format(trg))
			out_f.write('----------\n')

	translated_pred = list(map(lambda x: out_lang.EncodeAsPieces(x), translated_pred))
	translated_label = list(map(lambda x: out_lang.EncodeAsPieces(x), translated_label))

	min_pred_len = min([len(p) for p in translated_pred])
	min_label_len = min([len(p) for p in translated_label])
	log.debug('{} {}'.format(min_pred_len, min_label_len))
	log.debug('{}'.format(translated_pred[:4]))
	log.debug('{}'.format(translated_label[:4]))

	bleu = bleu_score(translated_pred, translated_label)
	log.info('bleu score: {:.4f}'.format(bleu))

	bleu = bleu_score(translated_pred, translated_pred)
	log.info('bleu score: {:.4f}'.format(bleu))

if __name__ == '__main__':
	main()






