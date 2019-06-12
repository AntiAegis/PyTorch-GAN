#------------------------------------------------------------------------------
#   Libraries
#------------------------------------------------------------------------------
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


#------------------------------------------------------------------------------
#   Loss function
#------------------------------------------------------------------------------
def loss_fn(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


#------------------------------------------------------------------------------
#  VAE
#------------------------------------------------------------------------------
class VAE(nn.Module):
	def __init__(self, in_dims=784, hid_dims=100, negative_slope=0.1):
		super(VAE, self).__init__()

		self.encoder = nn.Sequential(OrderedDict([
			('layer1', nn.Linear(in_dims, 512)),
			('relu1', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
			('layer2', nn.Linear(512, 256)),
			('relu2', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
			('layer3', nn.Linear(256, 128)),
			('relu3', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
		]))

		self.fc_mu = nn.Linear(128, hid_dims)
		self.fc_var = nn.Linear(128, hid_dims)

		self.decoder = nn.Sequential(OrderedDict([
			('layer1', nn.Linear(hid_dims, 128)),
			('relu1', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
			('layer2', nn.Linear(128, 256)),
			('relu2', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
			('layer3', nn.Linear(256, 512)),
			('relu3', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
			('layer4', nn.Linear(512, in_dims)),
			('sigmoid', nn.Sigmoid()),
		]))

		self._init_weights()

	def forward(self, x):
		h = self.encoder(x)
		mu, logvar = self.fc_mu(h), self.fc_var(h)
		z = self.reparameterize(mu, logvar)
		y = self.decoder(z)
		return y, mu, logvar

	def representation(self, x):
		h = self.encoder(x)
		mu, logvar = self.fc_mu(h), self.fc_var(h)
		z = self.reparameterize(mu, logvar)
		return z

	def reparameterize(self, mu, logvar):
		std = logvar.mul(0.5).exp_()
		esp = torch.randn(*mu.size()).cuda()
		z = mu + std * esp
		return z

	def _init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()


#------------------------------------------------------------------------------
#   ImproveChecker
#------------------------------------------------------------------------------
class ImproveChecker():
	def __init__(self, mode='min', best_val=None):
		assert mode in ['min', 'max']
		self.mode = mode
        if best_val is not None:
		    self.best_val = best_val
        else:
            if self.mode=='min':
                self.best_val = np.inf
            elif self.mode=='max':
                self.best_val = 0.0

	def check(self, val):
		if self.mode=='min':
			if val < self.best_val:
				print("[%s] Improved from %.4f to %.4f" % (self.__class__.__name__, val, self.best_val))
				self.best_val = val
				return True
			else:
				print("[%s] Not improved from %.4f to %.4f" % (self.__class__.__name__, val, self.best_val))
				return True
		else:
			if val > self.best_val:
				print("[%s] Improved from %.4f to %.4f" % (self.__class__.__name__, val, self.best_val))
				self.best_val = val
				return True
			else:
				print("[%s] Not improved from %.4f to %.4f" % (self.__class__.__name__, val, self.best_val))
				return True
