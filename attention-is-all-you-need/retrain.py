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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from setting import *
from feature import *
from optimizer import NoamOpt

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# define logger 
logging.config.fileConfig('logging.conf')
log = logging.getLogger('LaH')

fileHandler = logging.FileHandler('retrain.log')
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

'''
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
'''

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

def do_train(dataloader, model, loss_compute, epoch, lang):
	# change model to train mode
	model.train()

	start = time.time()
	total_tokens = 0
	total_loss = 0
	tokens = 0

	inp_lang, out_lang = lang
	with tqdm(dataloader, desc='training {}th epoch'.format(epoch)) as tbar:
		for i, batch in enumerate(tbar):
			# run model for training
			out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)

			# calculate loss
			loss = loss_compute(out, batch.trg, batch.ntokens)
			total_loss += loss
			total_tokens += batch.ntokens
			tokens += batch.ntokens

			# print loss
			if i % 50 == 1:
				elapsed = time.time() - start
				start = time.time()
				tokens = 0
		
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

			# run model for training
			out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)

			# calculate loss
			loss = loss_compute(out, batch.trg, batch.ntokens, do_backward=False)
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
	sent_pairs = load_dataset_aihub()
	#sent_pairs = load_dataset_aihub(path='data/')
	random.seed(100)
	random.shuffle(sent_pairs)
	log.debug('GPU#{} seeding with {}'.format(args.gpu, args.gpu))

	# make dataloader with dataset
	# FIXME: RuntimeError: Internal: unk is not defined.
	inp_lang, out_lang = get_sentencepiece(src_prefix, trg_prefix)
	log.info('loaded input sentencepiece model: {}'.format(src_prefix))
	log.info('loaded output sentencepiece model: {}'.format(trg_prefix))

	# split train/valid sentence pairs
	n_train = int(len(sent_pairs) * 0.8)
	n_split = int(n_train * 0.25)
	train_sent_pairs = sent_pairs[:n_train]
	log.info('train_sent_pairs: {}'.format(len(train_sent_pairs)))
	train_sent_pairs = train_sent_pairs[:args.gpu*n_split] + train_sent_pairs[(args.gpu+1)*n_split:]
	valid_sent_pairs = sent_pairs[n_train:]
	train_sent_pairs = sorted(train_sent_pairs, key=lambda x: (len(x[0]), len(x[1])))
	#log.info('train_sent_pairs: {}'.format(len(train_sent_pairs)))
	log.info('valid_sent_pairs: {}'.format(len(valid_sent_pairs)))

	# these are used for defining tokenize method and some reserved words
	SRC = KRENField(
		#tokenize=inp_lang.EncodeAsPieces, 
		pad_token='<pad>')
	TRG = KRENField(
		#tokenize=out_lang.EncodeAsPieces, 
		#init_token='<s>', 
		#eos_token='</s>', 
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
	#SRC.tokenize = src_tokenize
	#TRG.tokenize = trg_tokenize

	# make dataloader from KRENDataset
	train, valid, test = KRENDataset.splits(sent_pairs, (SRC, TRG), inp_lang, out_lang, encoding_type='pieces')
	# output -> ['<s>', '▁', 'Central', '▁Asian', '▁c', 'u', 'is', ... '▁yesterday', '.', '</s>']
	train_iter = MyIterator(train, batch_size=1024, device=0,
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

	# FIXME: need some good condition for multi distributed computing
	if True:
		model.cuda()

	model_name_full_path = './models/model-tmp.bin'
	device_pairs = zip([0], [args.gpu])                                                            
	map_location = {'cuda:{}'.format(x): '{}'.format('cuda:{}'.format(y)) for x, y in device_pairs}
	checkpoint = torch.load(model_name_full_path, map_location=map_location)
	state_dict = checkpoint['state_dict']
	model.load_state_dict(state_dict)
	model = DDP(model, device_ids=[args.gpu])

	# define model
	criterion = LabelSmoothing(size=args.out_n_words, padding_idx=0, smoothing=0.0)
	# FIXME: need some good condition for multi distributed computing
	if True:
		criterion.cuda()

	# define optimizer
	optimizer = NoamOpt(
			model_size=model.module.src_embed[0].d_model, 
			factor=1, 
			warmup=400,
			optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

	# initial best loss
	best_val_loss = checkpoint['best_val_loss']
	log.debug('best_val_loss: {:.4f}'.format(best_val_loss))

	# initialize visdom graph
	vis_train = Visdom()
	vis_valid = Visdom()

	train_loss_list = []
	valid_loss_list = []
	for epoch in range(args.epochs):
		train_losses = do_train((rebatch(pad_id, b) for b in train_iter),
				model,
				SimpleLossCompute(model.module.generator, criterion, opt=optimizer),
				epoch,
				(SRC, TRG))
		valid_loss = do_valid((rebatch(pad_id, b) for b in valid_iter),
				model,
				SimpleLossCompute(model.module.generator, criterion, opt=optimizer),
				epoch)

		if args.gpu == 0:
			if valid_loss >= best_val_loss:
				log.info('Try again. Current best is still {:.4f}'.format(best_val_loss))
			else:
				log.info('New record. from {:.4f} to {:.4f}'.format(best_val_loss, valid_loss))
				best_val_loss = valid_loss
				do_save(model, optimizer, epoch, best_val_loss)
	train_loss_list = np.array(train_loss_list)
	valid_loss_list = np.array(valid_loss_list)

	# draw visdom graph
	vis_train.line(Y=train_loss_list, X=np.arange(len(train_loss_list))*50)
	vis_valid.line(Y=valid_loss_list, X=np.arange(len(valid_loss_list)))

if __name__ == '__main__':
	main()


