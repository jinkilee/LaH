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

from torchtext.data.metrics import bleu_score
from utils import fix_torch_randomness, get_sentencepiece, to_gpu
from transformer import *
from dataset import load_dataset, set_padding, TranslationDataset
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
		print(i, out.shape, out[:,-1].shape)
		prob = model.generator(out[:, -1])
		print(prob.shape)
		print(prob)
		_, next_word = torch.max(prob, dim = 1)
		next_word = next_word.data[0]
		print('------')
		ys = torch.cat([ys, 
						torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
	return ys

def do_translate(dataloader, model, translate, device, epoch):
	# change model to validation mode
	model.eval()

	original_input = []
	translated_ids = []
	translated_lbl = []
	with tqdm(dataloader, desc='validating {}th epoch'.format(epoch)) as tbar:
		for i, batch in enumerate(tbar):
			# run model for training
			batch_src, batch_trg, batch_src_mask, batch_trg_mask, batch_ntokens = to_gpu(batch, device)
			out = greedy_decode(model, batch_src, batch_src_mask, max_len=60, start_symbol=bos_id)

			log.debug('x: {}'.format(batch_src))
			log.debug('p: {}'.format(out))
			log.debug('y: {}'.format(batch_trg))
			log.debug('-------------')

			original_input.extend(batch_src.to('cpu').numpy().tolist())
			translated_ids.extend(out.to('cpu').numpy().tolist())
			translated_lbl.extend(batch_trg.to('cpu').numpy().tolist())
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

parser = argparse.ArgumentParser()
parser.add_argument('--n_words', default=setting.n_words, type=int)
parser.add_argument('--gpu', default=setting.default_gpu, type=int)

def main():
	args = parser.parse_args()
	torch.cuda.set_device(args.gpu)

	# load dataset
	src = [oneline.rstrip() for oneline in open('kor_src.txt', 'r')]
	trg = [oneline.rstrip() for oneline in open('eng_trg.txt', 'r')]
	sent_pairs = [[s, t] for s, t in zip(src, trg)]

	# split train/valid sentence pairs
	n_train = int(len(sent_pairs) * 0.8)
	valid_sent_pairs = sent_pairs[n_train:]

	# make dataloader with dataset
	# FIXME: RuntimeError: Internal: unk is not defined.
	inp_lang, out_lang = get_sentencepiece(src_prefix, trg_prefix)
	valid_dataset = TranslationDataset(valid_sent_pairs, inp_lang, out_lang)
	valid_dataloader = DataLoader(valid_dataset, batch_size=1, collate_fn=set_padding)

	# fix torch randomness
	fix_torch_randomness()

	# FIXME: fix hard-coding
	args.inp_n_words = src_vocab_size
	args.out_n_words = trg_vocab_size
	model = make_model(args.inp_n_words, args.out_n_words, dropout=0.0)

	# Train the simple copy task.
	criterion = LabelSmoothing(size=args.n_words, padding_idx=0, smoothing=0.0)
	criterion.cuda()

	optimizer = NoamOpt(
			model.src_embed[0].d_model, 
			1, 
			400,
			torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

	# load model
	model_name_full_path = './models/model-tmp.bin'
	device_pairs = zip([0], [args.gpu])
	map_location = {'cuda:{}'.format(x): '{}'.format('cuda:{}'.format(y)) for x, y in device_pairs}
	checkpoint = torch.load(model_name_full_path, map_location=map_location)
	state_dict = checkpoint['state_dict']
	model.load_state_dict(state_dict)

	sum_of_weight = sum([p[1].data.sum() for p in model.named_parameters()])
	log.info('model was successfully loaded: {:.4f}'.format(sum_of_weight))

	# make gpu-distributed model
	device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
	model.to(device)

	original_input, translated_pred, translated_lbl = do_translate(valid_dataloader,
			model,
			SimpleTranslation(model.generator),
			device,
			0)

	print(original_input)
	original_input = list(map(lambda x: inp_lang.word2idx[x], original_input))
	translated_pred = list(map(lambda x: out_lang.word2idx[x], translated_pred))
	translated_label = list(map(lambda x: out_lang.word2idx[x], translated_lbl))

	with open('output/model-tmp.out', 'w', encoding='utf-8') as out_f:
		for src, pred, trg in zip(original_input, translated_pred, translated_label):
			src = ''.join(src)
			pred = ''.join(pred)
			trg = ''.join(trg)
			out_f.write('input: {}\n'.format(src))
			out_f.write('pred : {}\n'.format(pred))
			out_f.write('label: {}\n'.format(trg))
			out_f.write('----------\n')

	#translated_pred_text = list(map(lambda x: ''.join(x), translated_pred))
	#translated_label_text = list(map(lambda x: ''.join(x), translated_label))
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






