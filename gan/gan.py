#------------------------------------------------------------------------------
#  Libraries
#------------------------------------------------------------------------------
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

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
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=2e-4, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval betwen image samples')
parser.add_argument('--use_cuda', type=bool, default=True, help='use cuda to train model')
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)
cuda = True if (torch.cuda.is_available() and opt.use_cuda) else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


#------------------------------------------------------------------------------
#  Class of Generator
#------------------------------------------------------------------------------
class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()

		def block(in_feat, out_feat, normalize=True):
			layers = [nn.Linear(in_feat, out_feat)]
			if normalize:
				layers.append(nn.BatchNorm1d(out_feat, 0.8))
			layers.append(nn.LeakyReLU(0.2, inplace=True))
			return layers

		self.model = nn.Sequential(
			*block(opt.latent_dim, 128, normalize=False),
			*block(128, 256),
			*block(256, 512),
			*block(512, 1024),
			nn.Linear(1024, int(np.prod(img_shape))),
			nn.Tanh()
		)

	def forward(self, z):
		img = self.model(z)
		img = img.view(img.size(0), *img_shape)
		return img


#------------------------------------------------------------------------------
#  Class of Discriminator
#------------------------------------------------------------------------------
class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()

		self.model = nn.Sequential(
			nn.Linear(int(np.prod(img_shape)), 512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(512, 256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(256, 1),
			nn.Sigmoid()
		)

	def forward(self, img):
		img_flat = img.view(img.size(0), -1)
		validity = self.model(img_flat)

		return validity


#------------------------------------------------------------------------------
#  Setup
#------------------------------------------------------------------------------
# Loss function
adversarial_loss = nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# Configure data loader
data_dir = "/data4/livesegmentation/thuync/MNIST"
os.makedirs(data_dir, exist_ok=True)
dataset = datasets.MNIST(data_dir, train=True, download=True,
	transform=transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))
dataloader = torch.utils.data.DataLoader(
	dataset, batch_size=opt.batch_size,
	num_workers=opt.n_cpu, shuffle=True, pin_memory=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# TensorboardX
if os.path.exists("logging"):
	rmtree("logging")
writer_G = SummaryWriter(log_dir="logging/G")
# writer_G.add_graph(generator, torch.zeros(1, opt.latent_dim))

writer_D = SummaryWriter(log_dir="logging/D")
# writer_D.add_graph(discriminator, torch.zeros(1, *img_shape))

# Send to cuda
if cuda:
	generator.cuda()
	discriminator.cuda()
	adversarial_loss.cuda()


#------------------------------------------------------------------------------
#  Training
#------------------------------------------------------------------------------
for epoch in range(opt.n_epochs):
	for i, (imgs, _) in enumerate(dataloader):

		# Prepare input
		n_samples = imgs.shape[0]
		valid = torch.ones([n_samples, 1]).type(Tensor)
		fake = torch.zeros([n_samples, 1]).type(Tensor)
		real_imgs = imgs.type(Tensor)

		# -----------------
		#  Train Generator
		# -----------------
		generator.train()
		optimizer_G.zero_grad()
		discriminator.eval()

		# Sample noise as generator input
		z = Tensor(np.random.normal(0, 1, (n_samples, opt.latent_dim)))

		# Generate a batch of images
		gen_imgs = generator(z)

		# Loss measures generator's ability to fool the discriminator
		g_loss = adversarial_loss(discriminator(gen_imgs), valid)
		g_loss.backward()
		optimizer_G.step()

		# ---------------------
		#  Train Discriminator
		# ---------------------
		discriminator.train()
		optimizer_D.zero_grad()
		generator.eval()

		# Measure discriminator's ability to classify real from generated samples
		real_loss = adversarial_loss(discriminator(real_imgs), valid)
		fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
		d_loss = (real_loss + fake_loss) / 2

		d_loss.backward()
		optimizer_D.step()

		# ---------------------
		#  Logging
		# ---------------------
		batches_done = epoch * len(dataloader) + i
		if batches_done % opt.sample_interval == 0:
			grid = make_grid(gen_imgs.data[:25], nrow=5, normalize=True)
			writer_G.add_image('gen_image', grid, batches_done)

			writer_G.add_scalar("loss", g_loss.item(), batches_done)
			writer_D.add_scalar("loss", d_loss.item(), batches_done)

			# for name, param in discriminator.named_parameters():
			# 	writer_D.add_histogram("parameters/"+name, param.clone().cpu().data.numpy(), batches_done)
			# 	writer_D.add_histogram("gradients/"+name, param.grad.clone().cpu().data.numpy(), batches_done)

			print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (
				epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item()))
