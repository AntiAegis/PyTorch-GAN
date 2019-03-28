#------------------------------------------------------------------------------
#  Libraries
#------------------------------------------------------------------------------
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import numpy as np
from shutil import rmtree
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter


#------------------------------------------------------------------------------
#  Arguments
#------------------------------------------------------------------------------
# Data
NUM_EPOCH = 500
BATCHS_SIZE = 128
NUM_CPUS = 8

# Optimizer
LR = 1e-3

# Logging
LOG_DIR = "./logging"


#------------------------------------------------------------------------------
#  Class of VAE
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
#   Loss function
#------------------------------------------------------------------------------
def loss_fn(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


#------------------------------------------------------------------------------
#  Setup
#------------------------------------------------------------------------------
# Initialize VAE
model = VAE(in_dims=784, hid_dims=100)
model.cuda()

# Configure data loader
data_dir = "/media/antiaegis/storing/datasets/MNIST/"
os.makedirs(data_dir, exist_ok=True)
dataset = datasets.MNIST(data_dir, train=True, download=True,
	transform=transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))
dataloader = torch.utils.data.DataLoader(
	dataset, batch_size=BATCHS_SIZE,
	num_workers=NUM_CPUS, shuffle=True, pin_memory=True,
)

# Optimizers
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# TensorboardX
if os.path.exists(LOG_DIR):
	rmtree(LOG_DIR)
writer = SummaryWriter(log_dir=LOG_DIR)


#------------------------------------------------------------------------------
#  Training
#------------------------------------------------------------------------------
model.train()
for epoch in range(1, NUM_EPOCH+1):
	for i, (imgs, _) in enumerate(dataloader):
		# Prepare input
		inputs = imgs.view(imgs.shape[0], -1)
		inputs = inputs.cuda()

		# Train
		optimizer.zero_grad()
		outputs, mu, logvar = model(inputs)
		loss = loss_fn(outputs, inputs, mu, logvar)
		loss.backward()
		optimizer.step()

	# Logging
	# grid = make_grid(gen_imgs.data[:25], nrow=5, normalize=True)
	# writer.add_image('output', grid, epoch)
	writer.add_scalar("loss", loss.item(), epoch)
	print("[EPOCH %.3d] Loss: %.6f" % (epoch, loss.item()))