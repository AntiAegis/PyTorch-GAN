#------------------------------------------------------------------------------
#   Libraries
#------------------------------------------------------------------------------
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import numpy as np
from shutil import rmtree

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

from torchvision import datasets
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter


#------------------------------------------------------------------------------
#   Arguments
#------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=128, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=50, help='interval between image sampling')
parser.add_argument('--use_cuda', type=bool, default=True, help='use cuda to train model')
opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)
cuda = True if (torch.cuda.is_available() and opt.use_cuda) else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


#------------------------------------------------------------------------------
#   Weight initialization
#------------------------------------------------------------------------------
def weights_init_normal(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm2d') != -1:
		torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
		torch.nn.init.constant_(m.bias.data, 0.0)


#------------------------------------------------------------------------------
#   Generator
#------------------------------------------------------------------------------
class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()

		self.l1 = nn.Sequential(
			nn.Linear(opt.latent_dim, 32768),
			nn.BatchNorm1d(32768),
			nn.LeakyReLU(0.1, True)
		)
		self.conv_blocks = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
			nn.BatchNorm2d(128, momentum=0.9),
			nn.LeakyReLU(0.1, True),

			nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(128, momentum=0.9),
			nn.LeakyReLU(0.1, True),

			nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
			nn.BatchNorm2d(128, momentum=0.9),
			nn.LeakyReLU(0.1, True),

			nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
			nn.BatchNorm2d(128, momentum=0.9),
			nn.LeakyReLU(0.1, True),

			nn.Conv2d(128, opt.channels, kernel_size=5, stride=1, padding=2),
			nn.Tanh()
		)


	def forward(self, z):
		out = self.l1(z)
		out = out.view(out.shape[0], 128, 16, 16)
		img = self.conv_blocks(out)
		return img


#------------------------------------------------------------------------------
#   Discriminator
#------------------------------------------------------------------------------
class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()

		self.model = nn.Sequential(
			nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(128, momentum=0.9),
			nn.LeakyReLU(0.1, True),

			nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(128, momentum=0.9),
			nn.LeakyReLU(0.1, True),

			nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(128, momentum=0.9),
			nn.LeakyReLU(0.1, True),

			nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(128, momentum=0.9),
			nn.LeakyReLU(0.1, True),
		)

		self.final = nn.Sequential(
			nn.Dropout(0.4),
			nn.Linear(2048, 1),
			nn.Sigmoid()
		)

	def forward(self, img):
		out = self.model(img)
		out = out.view(out.shape[0], 2048)
		validity = self.final(out)
		return validity


#------------------------------------------------------------------------------
#   Setup
#------------------------------------------------------------------------------
# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
data_dir = "/media/antiaegis/storing/datasets/MNIST"
os.makedirs(data_dir, exist_ok=True)
dataset = datasets.MNIST(data_dir, train=True, download=True,
	transform=transforms.Compose([
		transforms.Resize(opt.img_size),
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
writer_D = SummaryWriter(log_dir="logging/D")

# Send to cuda
if cuda:
	generator.cuda()
	discriminator.cuda()
	adversarial_loss.cuda()


#------------------------------------------------------------------------------
#   Training
#------------------------------------------------------------------------------
for epoch in range(opt.n_epochs):
	for i, (imgs, _) in enumerate(dataloader):

		# ---------------------
		#  Prepare input
		# ---------------------
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
		# z = Tensor(np.random.normal(0, 1, (n_samples, opt.latent_dim)))
		z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

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