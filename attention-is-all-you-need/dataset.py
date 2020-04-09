import os
import re
import glob
import torch
import random
import string
import pandas as pd
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset
from torch.autograd import Variable
from setting import pad_id
from feature import *
from torch.nn.utils.rnn import pad_sequence
from torchtext.datasets.translation import TranslationDataset
from torchtext import data
from torchtext.vocab import Vocab
from collections import Counter

# perform basic cleaning
exclude = set(string.punctuation) # Set of all special characters
remove_digits = str.maketrans('', '', string.digits) # Set of all digits

MIN, MAX = 3, 60

def subsequent_mask(size):
	"Mask out subsequent positions."
	attn_shape = (1, size, size)
	subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
	return torch.from_numpy(subsequent_mask) == 0

class Batch:
	def __init__(self, src, trg=None, pad=0):
		self.src = src
		self.src_mask = (src != pad).unsqueeze(-2)
		if trg is not None:
			self.trg = trg[:, :-1]
			self.trg_y = trg[:, 1:]
			self.trg_mask = self.make_std_mask(self.trg, pad)
			self.ntokens = (self.trg_y != pad).data.sum()

	@staticmethod
	def make_std_mask(tgt, pad):
		tgt_mask = (tgt != pad).unsqueeze(-2)
		tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
		return tgt_mask

class KRENDataset(TranslationDataset):
	"""The IWSLT 2016 TED talk translation task"""

	def __init__(self, sent_pairs, fields, inp_lang, out_lang, encoding_type='ids'):
		"""Create a TranslationDataset given paths and fields.

		Arguments:
			fields: A tuple containing the fields that will be used for data
				in each language.
			Remaining keyword arguments: Passed to the constructor of
				data.Dataset.
		"""
		if not isinstance(fields[0], (tuple, list)):
			fields = [('src', fields[0]), ('trg', fields[1])]

		sent_pairs = list(filter(lambda x: len(x[0])*len(x[1]) != 0, sent_pairs))
		sent_pairs = list(filter(lambda x: MIN <= len(x[0]) and len(x[0]) <= MAX, sent_pairs))
		sent_pairs = list(filter(lambda x: MIN <= len(x[1]) and len(x[1]) <= MAX, sent_pairs))

		if encoding_type == 'ids':
			inp_encoding = inp_lang.EncodeAsIds
			out_encoding = out_lang.EncodeAsIds
		elif encoding_type == 'pieces':
			inp_encoding = inp_lang.EncodeAsPieces
			out_encoding = out_lang.EncodeAsPieces
		else:
			inp_encoding = None
			out_encoding = None

		examples = []
		#examples.append(data.Example.fromlist([src_line, trg_line], fields))
		for pair in tqdm(sent_pairs, desc='Encoding as pieces'):
			src, trg = pair
			pair = [
				inp_encoding(src),
				out_encoding(trg)
			]
			examples.append(data.Example.fromlist(pair, fields))

		super(TranslationDataset, self).__init__(examples, fields)

	@classmethod
	def splits(cls, sent_pairs, fields, inp_lang, out_lang, split_ratio=[0.8,0.1,0.1], **kwargs):
		"""Create dataset objects for splits of the IWSLT dataset.

		Arguments:
			exts: A tuple containing the extension to path for each language.
			fields: A tuple containing the fields that will be used for data
				in each language.
			root: Root dataset storage directory. Default is '.data'.
			train: The prefix of the train data. Default: 'train'.
			validation: The prefix of the validation data. Default: 'val'.
			test: The prefix of the test data. Default: 'test'.
			Remaining keyword arguments: Passed to the splits method of
				Dataset.
		"""
		assert sum(split_ratio) == 1, 'split_ratio should sum up to 1.0'

		split_ratio = np.cumsum(split_ratio)
		split_ratio = list(map(lambda x: int(len(sent_pairs)*x), split_ratio))
		n_train, n_valid, n_test = split_ratio
		train_sent_pairs = sent_pairs[0:n_train]
		valid_sent_pairs = sent_pairs[n_train:n_valid]
		test_sent_pairs = sent_pairs[n_valid:]

		train_data = cls(train_sent_pairs, fields, inp_lang, out_lang, **kwargs)
		val_data = cls(valid_sent_pairs, fields, inp_lang, out_lang, **kwargs)
		test_data = cls(test_sent_pairs, fields, inp_lang, out_lang, **kwargs)

		return tuple(d for d in (train_data, val_data, test_data)
					 if d is not None)

# This class creates a word -> index mapping (e.g,. "dad" -> 5) and vice-versa 
# (e.g., 5 -> "dad") for each language,
class LanguageIndex():
	def __init__(self, lang):
		self.lang = lang
		self.word2idx = {}
		self.idx2word = {}
		self.vocab = set()

		self.create_index()

	def create_index(self):
		for phrase in self.lang:
			self.vocab.update(phrase.split(' '))

		self.vocab = sorted(self.vocab)

		self.word2idx['<pad>'] = 0
		self.word2idx['<bos>'] = 1
		self.word2idx['<eos>'] = 2
		for index, word in enumerate(self.vocab):
			self.word2idx[word] = index + 3

		for word, index in self.word2idx.items():
			self.idx2word[index] = word

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
	longest_idx = np.argmax([trg_mask.size()[-1] for trg_mask in trg_mask_list])
	longest_mask_tensor = trg_mask_list[longest_idx]
	trg_mask_tensor = torch.cat([longest_mask_tensor]*len(trg_mask_list))

	return [src_token_tensor, trg_token_tensor, src_mask_tensor, trg_mask_tensor, ntokens]


def preprocess_kor_sentence(sent, add_bos_eos=False):
	'''Function to preprocess Marathi sentence'''
	sent = re.sub("'", '', sent)
	sent = ''.join(ch if ch not in exclude else ' ' for ch in sent)
	sent = sent.strip()
	sent = re.sub(" +", " ", sent)

	if add_bos_eos:
		sent = '<bos> ' + sent + ' <eos>'
	return sent

def preprocess_eng_sentence(sent, add_bos_eos=False):
	'''Function to preprocess English sentence'''
	sent = sent.lower()
	sent = re.sub("'", '', sent)
	sent = ''.join(ch if ch not in exclude else ' ' for ch in sent)
	sent = sent.translate(remove_digits)
	sent = sent.strip()
	sent = re.sub(" +", " ", sent)
	if add_bos_eos:
		sent = '<bos> ' + sent + ' <eos>'
	return sent

def load_dataset_aihub(path='../dataset/aihub', seed=100):
	sent_pairs = []
	for f in os.listdir(path):
		one_df = pd.read_excel(os.path.join(path, f))
		one_df = one_df.rename(columns={
			'영어':'eng',
			'한국어':'kor',
			'원문':'kor',
			'영어 검수':'label',
			'영어검수':'label',
			'번역문':'label',
			'Review':'label',
			'REVIEW':'label',
		})
		sent_pairs.extend(one_df[['kor','label']].values.tolist())
	random.seed(seed)
	random.shuffle(sent_pairs)
	return sent_pairs

def load_dataset_kaist(path):
	files = glob.glob(os.path.join(path, 'Corpus10', 'cekcorpus*.txt'))
	sent_pairs_list = []
	for corpus in files:
		kor_sents = []
		eng_sents = []
		idx = 0
		with open(corpus, 'r', encoding='cp949') as f:
			try:
				for i, oneline in enumerate(f):
					oneline = oneline.rstrip()
					if oneline.startswith('#'):
						oneline = oneline[1:]
					if i%4 == 0:
						idx += 1
					if i%4 == 1:
						eng_sents.append(oneline)
					if i%4 == 2:
						kor_sents.append(oneline)
			except UnicodeDecodeError as ue:
				print('Exception: {}th sentence of {}: {}'.format(idx, corpus, ue))
		kor_sents = list(map(preprocess_kor_sentence, kor_sents))
		eng_sents = list(map(preprocess_eng_sentence, eng_sents))

		if len(eng_sents) != len(kor_sents):
			continue
		sent_pairs = [[kor, eng] for kor, eng in zip(kor_sents, eng_sents)]
		if eng_sents[-1] == '':
			print('hello')
			print(corpus)
			exit()
		sent_pairs_list.extend(sent_pairs)
	return sent_pairs_list

def load_dataset_kaggle(path):
	# Generate pairs of cleaned English and Marathi sentences
	sent_pairs = []
	df = pd.read_csv(os.path.join(path, 'kaggle', 'kor_eng.csv'), encoding='cp949')
	for i in range(len(df)):
		sent_pair = []
		ko, en = df.iloc[i][['kor','eng']].values
		
		# append korean
		ko = preprocess_kor_sentence(ko)
		sent_pair.append(ko)
		
		# append english
		en = preprocess_kor_sentence(en)
		sent_pair.append(en)
		
		# append sentence pair
		sent_pairs.append(sent_pair)
	return sent_pairs

load_f_list = [
	load_dataset_kaist,
	load_dataset_kaggle,
]

def load_dataset(path='/home/jkfirst/workspace/git/LaH/dataset', seed=100):
	sent_pairs_list = []
	for load_f in load_f_list:
		sent_pairs_list.extend(load_f(path))
		print('>> {} sentence pairs loaded'.format(len(sent_pairs_list)))
	random.seed(seed)
	random.shuffle(sent_pairs_list)
	return sent_pairs_list

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
	longest_idx = np.argmax([trg_mask.size()[-1] for trg_mask in trg_mask_list])
	longest_mask_tensor = trg_mask_list[longest_idx]
	trg_mask_tensor = torch.cat([longest_mask_tensor]*len(trg_mask_list))

	return [src_token_tensor, trg_token_tensor, src_mask_tensor, trg_mask_tensor, ntokens]

def subsequent_mask(size):
	"Mask out subsequent positions."
	attn_shape = (1, size, size)
	subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
	return torch.from_numpy(subsequent_mask) == 0

class KRENField(data.Field):
	def build_vocab(self, dataset, spm_vocab_filename, **kwargs):
		'''
			This function has to make `self.vocab` which has the following properties.
			- freqs
			- itos
			- unk_index
			- stoi
			- vectors
		'''
		counter = Counter()

		for d in dataset:
			counter.update(d)
		self.vocab = Vocab(counter, specials=['<pad>','<s>','</s>','<unk>'])

		# edit itos
		itos = {}
		with open(spm_vocab_filename, 'r') as f:
			for line_num,line in enumerate(f):
				itos[line_num] = line.split("\t")[0]
		stoi = {v:k for k,v in itos.items()}
		self.vocab.itos = itos
		self.vocab.stoi = stoi

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
	src, trg = src.cuda(), trg.cuda()
	return Batch(src, trg, pad_idx)

