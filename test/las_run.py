import sys
sys.path.insert(0, './')

import torch
import numpy as np
from modeling.las_modeling import Listener, Speller

maxlen = 48
input_size = 80
listener_hidden_size = 160
speller_hidden_size = 2*listener_hidden_size
n_batch = 1
n_char = 30

encoder = Listener(input_size, listener_hidden_size, 3, 'LSTM', use_gpu=False)
decoder = Speller(
		output_class_dim=n_char, 
		speller_hidden_dim=speller_hidden_size, 
		rnn_unit='LSTM', 
		speller_rnn_layer=1, 
		use_gpu=False, 
		max_label_len=maxlen,
		use_mlp_in_attention=True,
		mlp_dim_in_attention=64,
		mlp_activate_in_attention='relu',
		listener_hidden_dim=listener_hidden_size,
		multi_head=1,
		decode_mode=1)

# make random input
inp = np.random.random((n_batch, maxlen, input_size))
inp = torch.FloatTensor(inp)
lbl = np.random.randint(0, n_char, (n_batch, maxlen))
print('label[0]: {}'.format(lbl[0,:10]))

# encoder output
# input:  (B, M, E)
# output: (B, M/8, H)
h = encoder(inp)
print('intermediate shape: {}'.format(h.shape))

# decoder output
raw_pred_seq, attention_record = decoder(h)
print('raw_pred_seq: {}'.format(len(raw_pred_seq)))
print('raw_pred_seq[0]: {}'.format(raw_pred_seq[0].shape))
print('attention_record: {}'.format(len(attention_record)))
#print('attention_record[0]: {}'.format(attention_record[0]))
print('attention_record[0][0]: {}'.format(attention_record[0][0].shape))
print('attention_record[47][0]: {}'.format(attention_record[-1][0].shape))



