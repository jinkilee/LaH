import torch
from torch import nn
from torch.autograd import Variable

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

class MultiGPULossCompute:
	"A multi-gpu loss compute and train function."
	def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
		# Send out to different gpus.
		self.generator = generator
		self.criterion = nn.parallel.replicate(criterion,
											   devices=devices)
		self.opt = opt
		self.devices = devices
		self.chunk_size = chunk_size
		self.requires_grad = self.opt is not None

	def __call__(self, out, targets, normalize):
		total = 0.0
		#print('requires_grad: ', self.requires_grad)
		generator = nn.parallel.replicate(self.generator, devices=self.devices)
		out_scatter = nn.parallel.scatter(out,
										  target_gpus=self.devices)
		out_grad = [[] for _ in out_scatter]
		targets = nn.parallel.scatter(targets,
									  target_gpus=self.devices)

		# Divide generating into chunks.
		chunk_size = self.chunk_size
		for i in range(0, out_scatter[0].size(1), chunk_size):
			# Predict distributions
			out_column = [[Variable(o[:, i:i+chunk_size].data,
									requires_grad=self.opt is not None)]
						   for o in out_scatter]
			gen = nn.parallel.parallel_apply(generator, out_column)

			# Compute loss.
			y = [(g.contiguous().view(-1, g.size(-1)),
				  t[:, i:i+chunk_size].contiguous().view(-1))
				 for g, t in zip(gen, targets)]
			loss = nn.parallel.parallel_apply(self.criterion, y)

			# Sum and normalize loss
			l = nn.parallel.gather(loss,
								   target_device=self.devices[0])
			#l1 = l.sum().item() / normalize
			l2 = l.sum() / normalize
			total += l2.data.item()

			# Backprop loss to output of transformer
			if self.opt is not None:
				#l.backward()
				#l2.sum().backward()
				l2.backward()
				for j, l in enumerate(loss):
					out_grad[j].append(out_column[j][0].grad.data.clone())

		# Backprop all loss through transformer.
		if self.opt is not None:
			out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
			o1 = out
			o2 = nn.parallel.gather(out_grad,
									target_device=self.devices[0])
			o1.backward(gradient=o2)
			self.opt.step()
			self.opt.optimizer.zero_grad()
			
		return total * normalize.float()


class MultiGPULossCompute_(object):
	def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
		self.opt = opt
		self.devices = devices
		self.chunk_size = chunk_size

		self.criteria = [nn.parallel.replicate(criterion, devices=devices[:i+1]) for i in range(len(devices))]
		self.generators = [nn.parallel.replicate(generator, devices=devices[:i+1]) for i in range(len(devices))]

	def __call__(self, out, target, normalize):
		total = 0.0
		out_scatter = nn.parallel.scatter(out, target_gpus=self.devices)
		out_grad = [[] for _ in out_scatter]
		targets = nn.parallel.scatter(target, target_gpus=self.devices)

		generator = self.generators[len(out_scatter) - 1]
		criterion = self.criteria[len(out_scatter) - 1]

		# Divide generating into chunks.
		chunk_size = self.chunk_size
		for i in range(0, out_scatter[0].size(1), chunk_size):
			# Predict distributions
			out_column = [[Variable(o[:, i:i+chunk_size].data,
									requires_grad=self.opt is not None)]
						   for o in out_scatter]
			gen = nn.parallel.parallel_apply(generator, out_column)

			# Compute loss.
			y = [(g.contiguous().view(-1, g.size(-1)),
				  t[:, i:i+chunk_size].contiguous().view(-1))
				 for g, t in zip(gen, targets)]
			loss = nn.parallel.parallel_apply(criterion, y)

			# Sum and normalize loss
			l = nn.parallel.gather(loss,
								   target_device=self.devices[0])
			#l1 = l.sum().item() / normalize
			l2 = l.sum() / normalize
			total += l2.data.item()

			# Backprop loss to output of transformer
			if self.opt is not None:
				#l.backward()
				#l2.sum().backward()
				l2.backward()
				for j, l in enumerate(loss):
					out_grad[j].append(out_column[j][0].grad.data.clone())

		# Backprop all loss through transformer.
		if self.opt is not None:
			out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
			o1 = out
			o2 = nn.parallel.gather(out_grad,
									target_device=self.devices[0])
			o1.backward(gradient=o2)
			self.opt.step()
			self.opt.optimizer.zero_grad()
			
		return total * normalize.float()

