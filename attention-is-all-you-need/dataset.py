import os
import re
import glob
import torch
import random
import string
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from torch.autograd import Variable
from setting import pad_id
from feature import *
from torch.nn.utils.rnn import pad_sequence
from torchtext.datasets.translation import TranslationDataset
from torchtext import data

# perform basic cleaning
exclude = set(string.punctuation) # Set of all special characters
remove_digits = str.maketrans('', '', string.digits) # Set of all digits

MIN, MAX = 3, 30

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

	def __init__(self, sent_pairs, fields, inp_lang, out_lang):
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
		sent_pairs = list(filter(lambda x: len(x[0]) > MIN and len(x[1]) > MIN, sent_pairs))
		sent_pairs = list(filter(lambda x: len(x[0]) < MAX and len(x[1]) < MAX, sent_pairs))

		examples = []
		#examples.append(data.Example.fromlist([src_line, trg_line], fields))
		for pair in sent_pairs:
			src, trg = pair
			pair = [inp_lang.EncodeAsPieces(src), out_lang.EncodeAsPieces(trg)]
			examples.append(data.Example.fromlist(pair, fields))

		super(TranslationDataset, self).__init__(examples, fields)

	@classmethod
	def splits(cls, sent_pairs, fields, inp_lang, out_lang, **kwargs):
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

		n_train = int(len(sent_pairs) * 0.8)
		n_valid = int(len(sent_pairs) * 0.15)
		train_sent_pairs = sent_pairs[:n_train]
		valid_sent_pairs = sent_pairs[n_train:n_train+n_valid]
		test_sent_pairs = sent_pairs[n_train+n_valid:]

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

