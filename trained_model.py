# -*- coding: utf-8 -*-
"""
@author: berna
"""
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import scipy.io

#%%
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
#%%
class DatasetTest(Dataset):
    def __init__(self, list_data):
        images = []
        labels = []
        for img,lab in list_data:
            images.append(img)
            labels.append(lab)
        
        self.images = images
        self.labels = labels
        self.length = len(images)

    def __getitem__(self, index):
        sample = self.images[index]
        label = self.labels[index]
        
        tensor_sample = transforms.functional.to_tensor(sample).float()
        tensor_sample = tensor_sample.transpose(1, 2)
        return (tensor_sample, label)

    def __len__(self):
        return self.length 

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
#%%

### Testing function
def test_epoch(net, dataloader, loss_fn):
    
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

#%%
# load the dataset
mat = scipy.io.loadmat('MNIST.mat')
m_image = mat["input_images"];
m_labels = mat["output_labels"];
#%% test on sample from provided MNIST
#img = x_all[56]
#azz = np.resize(img*255,(28,28))
#fig, axs = plt.subplots(1, 2, figsize=(12,6))
#axs[0].imshow(azz)
#axs[0].set_title('Original image')
#%%

# Load the network to test
encoded_space_dim = 8
net = Autoencoder(encoded_space_dim=encoded_space_dim)
net.load_state_dict(torch.load('net_params_final_cross_valid.pth', map_location=device))
net.to(device)

#%%
test_data = []
for i in range(len(m_image)):
    
   bw_scale = np.asarray(np.resize(m_image[i] * 255, (28,28)),dtype = np.int8)
   image_array = Image.fromarray(bw_scale, mode='L')
   test_data.append([image_array.convert("L"),
                     int(m_labels[i].item())]);

test_data = DatasetTest(test_data)

test_dataloader = DataLoader(test_data, batch_size=256, shuffle=False)
loss_fn = torch.nn.MSELoss()

val_loss = test_epoch(net, dataloader=test_dataloader, loss_fn=loss_fn) 
print('\n\n\t TEST - loss: %f\n\n' % (val_loss))

#%% Plot progress

#img = test_data[1000][0].unsqueeze(0).to(device)
#
#net.eval()
#with torch.no_grad():
#    rec_img  = net(img)
#fig, axs = plt.subplots(1, 2, figsize=(12,6))
#axs[0].imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
#axs[0].set_title('Original image')
#axs[1].imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
#axs[1].set_title('Reconstructed image')
#plt.show()
#plt.close()























