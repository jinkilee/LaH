import os
import re
import glob
import string
import pandas as pd

# perform basic cleaning
exclude = set(string.punctuation) # Set of all special characters
remove_digits = str.maketrans('', '', string.digits) # Set of all digits

def preprocess_kor_sentence(sent):
	'''Function to preprocess Marathi sentence'''
	sent = re.sub("'", '', sent)
	sent = ''.join(ch for ch in sent if ch not in exclude)
	sent = sent.strip()
	sent = re.sub(" +", " ", sent)
	sent = '<start> ' + sent + ' <end>'
	return sent

def preprocess_eng_sentence(sent):
	'''Function to preprocess English sentence'''
	sent = sent.lower()
	sent = re.sub("'", '', sent)
	sent = ''.join(ch for ch in sent if ch not in exclude)
	sent = sent.translate(remove_digits)
	sent = sent.strip()
	sent = re.sub(" +", " ", sent)
	sent = '<start> ' + sent + ' <end>'
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

def load_dataset(path='/home/jkfirst/workspace/git/LaH/dataset'):
	sent_pairs_list = []
	for load_f in load_f_list:
		sent_pairs_list.extend(load_f(path))
		print('>> {} sentence pairs loaded'.format(len(sent_pairs_list)))
	return sent_pairs_list

#sent_pairs = load_dataset()




