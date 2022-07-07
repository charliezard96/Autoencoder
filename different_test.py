# -*- coding: utf-8 -*-
"""

@author: berna
"""

import torch
import matplotlib.pyplot as plt
import random
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import numpy as np

#%% Define paths

data_root_dir = '../datasets'


#%% function for gaussian noise
def gaussian(image, std): 
    row,col= image.shape
    mean = 0
    gauss = np.random.normal(mean,std,(row,col))
    gauss = gauss.reshape(row,col)
    res = image + gauss
    fin = transforms.functional.to_tensor(res).float()
    return fin
#%% Create dataset
p = 0
p1 = 0

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomErasing(p=p1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
    transforms.RandomApply([
        transforms.Lambda(lambda x: gaussian(x.squeeze().numpy(), 0.5)), 
        ], 
        p=1),
    transforms.RandomApply([
        transforms.Lambda(lambda x: gaussian(x.squeeze().numpy(), 0.05)), 
        ], 
        p=p1),
    transforms.RandomApply([
        transforms.Lambda(lambda x: gaussian(x.squeeze().numpy(), 0.01)), 
        ], 
        p=p1),        
])    

test_dataset  = MNIST(data_root_dir, train=False, download=True, transform=test_transform)

### Plot some test sample
plt.close('all')
fig, axs = plt.subplots(5, 5, figsize=(8,8))
for ax in axs.flatten():
    img, label = random.choice(test_dataset)
    ax.imshow(img.squeeze().numpy(), cmap='gist_gray')
    ax.set_title('Label: %d' % label)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()

#%% Define the network architecture
    
class Autoencoder(nn.Module):
    
    def __init__(self, encoded_space_dim):
        super().__init__()
        
        ### Encoder
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 64),
            nn.ReLU(True),
            nn.Linear(64, encoded_space_dim)
        )
        
        ### Decoder
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 3 * 3 * 32),
            nn.ReLU(True)
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        # Apply convolutions
        x = self.encoder_cnn(x)
        # Flatten
        x = x.view([x.size(0), -1])
        # Apply linear layers
        x = self.encoder_lin(x)
        return x
    
    def decode(self, x):
        # Apply linear layers
        x = self.decoder_lin(x)
        # Reshape
        x = x.view([-1, 32, 3, 3])
        # Apply transposed convolutions
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

### Initialize the network
encoded_space_dim = 6
net = Autoencoder(encoded_space_dim=encoded_space_dim)


### Some examples
# Take an input image (remember to add the batch dimension)
img = test_dataset[0][0].unsqueeze(0)
print('Original image shape:', img.shape)
# Encode the image
img_enc = net.encode(img)
print('Encoded image shape:', img_enc.shape)
# Decode the image
dec_img = net.decode(img_enc)
print('Decoded image shape:', dec_img.shape)
#%% Prepare test

### Define dataloader
test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False)

### Define a loss function
loss_fn = torch.nn.MSELoss()

### Define an optimizer
lr = 1e-3 # Learning rate
optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
#%%
### Testing function
def test_epoch(net, dataloader, loss_fn, optimizer):
    # Validation
    net.eval() # Evaluation mode (e.g. disable dropout)
    with torch.no_grad(): # No need to track the gradients
        conc_out = torch.Tensor().float()
        conc_label = torch.Tensor().float()
        for sample_batch in dataloader:
            # Extract data and move tensors to the selected device
            image_batch = sample_batch[0].to(device)
            # Forward pass
            out = net(image_batch)
            # Concatenate with previous outputs
            conc_out = torch.cat([conc_out, out.cpu()])
            conc_label = torch.cat([conc_label, image_batch.cpu()]) 
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data

#%% Testing
encoded_space_dim = 8
cond = True
if (cond):
    # Load the network to test
    net = Autoencoder(encoded_space_dim=encoded_space_dim)
    net.load_state_dict(torch.load('net_params_final_cross_valid.pth', map_location=device))
    net.to(device)
    
num_test = random.randint(0, 1000)
img = test_dataset[num_test][0].unsqueeze(0).to(device)
label = test_dataset[num_test][1]
net.eval()
with torch.no_grad():
    rec_img  = net(img)
fig, axs = plt.subplots(1, 2, figsize=(12,6))
axs[0].imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
axs[0].set_title('Original image, Label: %d' % label)
axs[1].imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
axs[1].set_title('Reconstructed image (final)')
plt.tight_layout()
plt.pause(0.1)

final_test_loss= test_epoch(net, dataloader=test_dataloader, loss_fn=loss_fn, optimizer=optim) 
print('\n\n\t TEST - loss: %f\n\n' % (final_test_loss))