import torch
from torch import nn

class Encoder(nn.Module):
	def __init__(self, vocab_size, embedding_size, hidden_size, n_batch):
		super(Encoder, self).__init__()
		self.embedding_size = embedding_size
		self.hidden_size = hidden_size
		self.embedding = nn.Embedding(vocab_size, self.embedding_size)
		self.gru = nn.GRU(embedding_size, 
						  hidden_size,
						  num_layers=1,
						  bias=True,
						  batch_first=True,
						  bidirectional=False)
		self.init_hidden = torch.zeros(1, n_batch, self.hidden_size).cuda()
		
	def forward(self, x):
		x = self.embedding(x)
		output, hidden = self.gru(x, self.init_hidden)
		return output, hidden

class Decoder(nn.Module):
	def __init__(self, vocab_size, embedding_size, hidden_size, n_batch):
		super(Decoder, self).__init__()
		self.hidden_size = hidden_size
		self.embedding = nn.Embedding(vocab_size, embedding_size)
		self.gru = nn.GRU(embedding_size+hidden_size,
						 hidden_size,
						 num_layers=1,
						 bias=True, 
						 batch_first=True,
						 bidirectional=False)
		self.init_hidden = torch.zeros(1, n_batch, self.hidden_size).cuda()
		self.tahn = nn.Tanh()
		
		self.W1 = nn.Linear(self.hidden_size, self.hidden_size)
		self.W2 = nn.Linear(self.hidden_size, self.hidden_size)
		self.V = nn.Linear(self.hidden_size, 1)
		self.softmax = nn.Softmax(dim=1)
		self.fc = nn.Linear(self.hidden_size, vocab_size)
		
	def forward(self, x, hidden, enc_output):
		score = self.V(self.tahn(self.W1(enc_output) + self.W2(hidden)))
		attention_weights = self.softmax(score)
		context_vector = attention_weights * enc_output
		context_vector = context_vector.sum(dim=1)
		
		x = self.embedding(x)
		x = torch.cat((x, context_vector.unsqueeze(dim=1)), dim=-1)
		output, state = self.gru(x, self.init_hidden)
		output = output.squeeze(dim=1)
		output = self.fc(output)
		return output, state, attention_weights


class EncoderDecoder(nn.Module):
	def __init__(self, embedding_size, hidden_size, src_vocab_size, dst_vocab_size, n_batch):
		super(EncoderDecoder, self).__init__()
		
		self.encoder = Encoder(src_vocab_size, embedding_size, hidden_size, n_batch=n_batch)
		self.decoder = Decoder(dst_vocab_size, embedding_size, hidden_size, n_batch=n_batch)
		self.loss_fn = DecoderLoss()
		
	def forward(self, src, dst, start_token, n_batch):
		enc_output, enc_hidden = self.encoder(src)

		loss = 0
		hidden = enc_hidden.transpose(0, 1)
		dec_input = torch.LongTensor([start_token] * n_batch).to('cuda')
		#dec_input = torch.LongTensor([vocab(['<start>'])] * n_batch).to('cuda')
		for t in range(1, dst.size()[1]):
			output, state, _ = self.decoder(dec_input, hidden, enc_output)
			loss += self.loss_fn(dst[:, t], output)
		return loss


class DecoderLoss(nn.Module):
	def __init__(self):
		super(DecoderLoss, self).__init__()
		self.loss_fn = nn.CrossEntropyLoss()

	def forward(self, real, pred):
		mask = 1 - (real == 0).type(torch.LongTensor)
		loss = self.loss_fn(pred, real) * mask
		loss = loss.sum()
		return loss
