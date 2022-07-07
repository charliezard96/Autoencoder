# -*- coding: utf-8 -*-
"""
@author: berna
"""
import os
import torch
import matplotlib.pyplot as plt
import random
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
import numpy as np
from torch.utils.data.dataset import Subset

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

 #%%
    # creation of the k-fold (k=3)
def create_kFold(dataset, batch_size):
        
    len_data = int(len(dataset))
    i_list = list(np.arange(0,len(dataset)))
    
    vali1 = DataLoader(Subset(dataset, i_list[0:int(len_data/3)]), 
                       batch_size=batch_size, shuffle=True);

    vali2 = DataLoader(Subset(dataset, i_list[int(len_data/3): int(len_data/3)*2]), 
                       batch_size=batch_size, shuffle=True);

    vali3 = DataLoader(Subset(dataset, i_list[int(len_data/3)*2:len_data]), 
                       batch_size=batch_size, shuffle=True);
    
    tran1 = DataLoader(Subset(dataset, i_list[0:int(len_data/3)]), 
                       batch_size=batch_size, shuffle=True)
    
    tran3 = DataLoader(Subset(dataset, i_list[0:int(len_data/3)*2]), 
                       batch_size=batch_size, shuffle=True)
    
    i_list_for2 = i_list[0:int(len_data/3)] + i_list[int(len_data/3)*2:len_data]
    tran2 = DataLoader(Subset(dataset, i_list_for2), 
                       batch_size=batch_size, shuffle=True)
    kFold_dataset = [
            [tran1, vali1],  
            [tran2, vali2], 
            [tran3,vali3]
            ]
    
    return kFold_dataset
#%% Create dataset

train_transform = transforms.Compose([
    transforms.ToTensor(),
])
    
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.33, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
    transforms.RandomApply([
        transforms.Lambda(lambda x: gaussian(x.squeeze().numpy(), 0.1)), 
        ], 
        p=0.5),
    transforms.RandomApply([
        transforms.Lambda(lambda x: gaussian(x.squeeze().numpy(), 0.05)), 
        ], 
        p=1),
    transforms.RandomApply([
        transforms.Lambda(lambda x: gaussian(x.squeeze().numpy(), 0.5)), 
        ], 
        p=0.15),        
])    

train_dataset = MNIST(data_root_dir, train=True,  download=True, transform=test_transform)
test_dataset  = MNIST(data_root_dir, train=False, download=True, transform=train_transform)
#%%
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
#%%
### Plot some train sample
plt.close('all')
fig, axs = plt.subplots(5, 5, figsize=(8,8))
for ax in axs.flatten():
    img, label = random.choice(train_dataset)
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


#%% Prepare training

### Define dataloader
train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
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

#%% Network training

### Training function
def train_epoch(net, dataloader, loss_fn, optimizer):
    # Training
    net.train()
    loss_vec = []
    for sample_batch in dataloader:
        # Extract data and move tensors to the selected device
        image_batch = sample_batch[0].to(device)
        # Forward pass
        output = net(image_batch)
        loss = loss_fn(output, image_batch)
        # Backward pass
        optim.zero_grad()
        loss.backward()
        optim.step()
        loss_vec.append(loss)
        # Print loss
        #print('\t partial train loss: %f' % (loss.data))
    return torch.mean(torch.stack(loss_vec))

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

#%% training
    
best_param = [8, 1e-03, 256]

opt_coding_dim = best_param[0]
opt_lrate = best_param[1]
opt_batch_size = best_param[2]

net = Autoencoder(opt_coding_dim)
net.to(device)
optim = torch.optim.Adam(net.parameters(), lr=opt_lrate,  weight_decay=1e-5)

train_dataloader = DataLoader(train_dataset, batch_size=opt_batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=opt_batch_size, shuffle=False)

train_record = []
test_record = []
num_epochs_final = 200
last_loss_improve = 1000
last_improve = 0
for epoch in range(num_epochs_final):
        print('EPOCH %d/%d' % (epoch + 1, num_epochs_final))
        train_loss = train_epoch(net, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optim) 
        val_loss = test_epoch(net, dataloader=test_dataloader, loss_fn=loss_fn, optimizer=optim) 
        print('\n\n\t VALIDATION - EPOCH %d/%d - loss: %f\n\n' % (epoch + 1, num_epochs_final, val_loss))
        train_record.append(train_loss)
        test_record.append(val_loss)
        ### Plot progress
        img = test_dataset[0][0].unsqueeze(0).to(device)
        net.eval()
        with torch.no_grad():
            rec_img  = net(img)
        fig, axs = plt.subplots(1, 2, figsize=(12,6))
        axs[0].imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
        axs[0].set_title('Original image')
        axs[1].imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
        axs[1].set_title('Reconstructed image (EPOCH %d)' % (epoch + 1))
        plt.tight_layout()
        plt.pause(0.1)
        # Save figures
        os.makedirs('autoencoder_progress_%d_features' % encoded_space_dim, exist_ok=True)
        plt.savefig('autoencoder_progress_%d_features/epoch_%d.png' % (encoded_space_dim, epoch + 1))
        plt.show()
        plt.close()
        
        # Save network parameters
        torch.save(net.state_dict(), 'optional_net_params1.pth')
        
        if(last_loss_improve < val_loss):
            last_improve = last_improve + 1
        else:
            last_improve = 0
            last_loss_improve = val_loss
        
        if(last_improve > 30):
            print("end epoch: " + str(epoch))
            break

#%% Final loss

final_test_loss= test_epoch(net, dataloader=test_dataloader, loss_fn=loss_fn, optimizer=optim) 
print('\n\n\t TEST - loss: %f\n\n' % (final_test_loss))

#%% Losses

### Plot losses
plt.close('all')
plt.figure(figsize=(10,6))
plt.semilogy(train_record, label='Train loss')
plt.semilogy(test_record, label='Test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

#%% Testing on noise images
if (False):
    # Load the network to test
    encoded_space_dim = 6
    net = Autoencoder(encoded_space_dim=encoded_space_dim)
    net.load_state_dict(torch.load('optional_net_params.pth', map_location=device))
    net.to(device)
    
num_test = random.randint(0, 1000)
img = train_dataset[num_test][0].unsqueeze(0).to(device)
label = train_dataset[num_test][1]
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

#%% Testing on no-noise images
if (False):
    # Load the network to test
    encoded_space_dim = 6
    net = Autoencoder(encoded_space_dim=encoded_space_dim)
    net.load_state_dict(torch.load('optional_net_params.pth', map_location=device))
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