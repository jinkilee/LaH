import sys
sys.path.insert(0, './')

import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from gluonnlp.data import SentencepieceTokenizer
from modeling.seq2seq_modeling import EncoderDecoder
from transformers import BertModel, BertTokenizer
from kogpt2.utils import get_tokenizer
from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from torch.utils.data import Dataset, DataLoader

embedding_size = 10
hidden_size = 16
n_batch = 8

tok_path = get_tokenizer()
kor_tokenizer = SentencepieceTokenizer(tok_path)
_, vocab = get_pytorch_kogpt2_model()

model_nm = 'bert-large-uncased'
tokenizer = BertTokenizer.from_pretrained(model_nm)
eng_tokenizer = tokenizer.tokenize

def convert_string_to_index(kr=None, en=None, kr_pad=tokenizer.vocab['[PAD]'], en_pad=vocab(['<pad>'])[0], maxlen=50):
    '''
        convert korean/english sentence into its own indices.
        maximum length of converted indices is `maxlen`
    '''
    assert (kr != None) or (en != None), 'one of either kr or en should have a value'
    kr_index, en_index = None, None
    
    if kr:
        kr_index = vocab(kor_tokenizer(kr))
        if len(kr_index) > maxlen:
            kr_index = kr_index[:maxlen]
        else:
            kr_index = kr_index + [kr_pad] * (maxlen-len(kr_index))
    if en:
        en_index = tokenizer.convert_tokens_to_ids(eng_tokenizer(en))
        if len(en_index) > maxlen:
            en_index = en_index[:maxlen]
        else:
            en_index = en_index + [en_pad] * (maxlen-len(en_index))
        
    return kr_index, en_index

class TranslationText():
    def __init__(self, kr, en):
        self.kr = kr
        self.en = en

class TranslationDataset(Dataset):
    """Translation dataset."""

    def __init__(self, csv_file, transform=None, names=['kor', 'eng'], sep='\t'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            names (list): A list of column names
            sep (string): A string that is used for a delimiter.
        """
        #self.landmarks_frame = pd.read_csv(csv_file)
        self.df = pd.read_csv(csv_file, names=names, sep=sep)
        self.df = self.df[(self.df.kor.notnull()) & (self.df.eng.notnull())]
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        onerow =  self.df.iloc[idx]
        kr_index, en_index = convert_string_to_index(kr=onerow.kor, en=onerow.eng)
        kor_tensor = torch.LongTensor(kr_index)
        eng_tensor = torch.LongTensor(en_index)
        return (kor_tensor, eng_tensor)

data_dir = 'korean-parallel-corpora/korean-english-news-v1'
dataset = TranslationDataset(os.path.join(data_dir, 'train.txt'))
trainloader = DataLoader(dataset, batch_size = n_batch, shuffle = True)
targets = ['train', 'test', 'dev']

for target in targets:
    output_filename = '{}/{}.txt'.format(data_dir, target)
    if os.path.exists(output_filename):
        print('already have {}'.format(output_filename))
        continue
        
    ko_file, en_file = glob.glob('{}/*{}.??'.format(data_dir, target))
    
    # read korean/english files
    ko_lines = open(ko_file).read().strip().split('\n')
    en_lines = open(en_file).read().strip().split('\n')
    
    # write to output file
    with open(output_filename, 'w') as out:
        for en, kr in zip(en_lines, ko_lines):
            oneline = '\t'.join([en, kr])
            out.write(oneline + '\n')
            
    print('{} was written'.format(output_filename))

seq2seq = EncoderDecoder(
	embedding_size=embedding_size,
	hidden_size=hidden_size,
	src_vocab_size=tokenizer.vocab_size,
	dst_vocab_size=len(vocab),
	n_batch=n_batch)
seq2seq.cuda()
seq2seq.train()

n_epoch = 5
for epoch in range(n_epoch):
    total_loss = 0
    
    tbar = tqdm(enumerate(trainloader), desc='training at {}th epoch'.format(epoch))
    for i, (input_kr, input_en) in tbar:
        input_en = input_en.cuda()
        input_kr = input_kr.cuda()
        loss = seq2seq(
			input_en, 
			input_kr, 
			start_token=vocab(['<start>']), 
			n_batch=n_batch)
        
        # calculate loss
        batch_loss = loss / input_kr.size()[1]
        total_loss += batch_loss
        tbar.set_postfix(loss=batch_loss)
        
    print('Loss at {}th epoch: {:.4f}'.format(epoch, total_loss / n_batch))



