from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import  numpy as np
import itertools

from model_custom import ClientDiscriminator,ClientEncoder,ClientDecoder,weights_init,ClientClassifier,ClientGenerator
from utils import *

manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

print(device)

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))


# Create the generator
netE = ClientEncoder().to(device)
netDe = ClientDecoder().to(device)

# Handle multi-gpu if desired

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
#netG.apply(weights_init)

# Print the model



# Create the Discriminator
netD = ClientDiscriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
#netD.apply(weights_init)

# Print the model
print(netD)


# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerE = optim.Adam(netE.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerDe = optim.Adam(netDe.parameters(), lr=lr, betas=(beta1, 0.999))

adversarial_loss = torch.nn.BCELoss()
pixelwise_loss = torch.nn.L1Loss()
mae_loss = torch.nn.MSELoss()

adversarial_loss.to(device)
pixelwise_loss.to(device)
criterion.to(device)


# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0


print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader


    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch

        netE.zero_grad()
        netDe.zero_grad()
        netD.zero_grad()

        # Format batch
        real_img = data[0].to(device)
        b_size = real_img.size(0)

        label = torch.full((b_size,), real_label, dtype=torch.float, device=device).to(device)
        odd_label = torch.full((b_size,), fake_label, dtype=torch.float, device=device).to(device)

        
        encoder_img = netE(real_img)
        decoder_img = netDe(encoder_img)


        # Forward pass generate batch through De
        errRecons = mae_loss(real_img,decoder_img)

        #Calculate the gradients for generated img
        errRecons.backward()

        optimizerDe.step()
        optimizerE.step()

        noise = torch.randn(tuple(encoder_img.size()), device=device)


        # Forward pass  through Discriminator
        output_real = netD(noise).view(-1)
        output_fake = netD(encoder_img).view(-1)
        # Calculate loss on all-real batch

        #@print("all s",netD(noise).size(),label.size())
        errD_real = 0.001 * adversarial_loss(output_real, label)

        errD_fake = 0.001 * adversarial_loss(output_fake.detach(), odd_label)

        errD = errD_real + errD_fake

        # Calculate gradients for D in backward pass
        errD.backward(retain_graph=True)

        optimizerD.step()

        del label
        del odd_label
        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errRecons.item(), errD))



        # Save Losses for plotting later
        G_losses.append(errRecons.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                encoder_track = netE(real_batch[0].to(device)[:64])
                fake = netDe(encoder_track).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            plt.subplot(1, 2, 2)
            plt.axis("off")
            plt.title("Fake Images")
            plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
            plt.show()

        iters += 1



plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


#%%capture
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())


real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()