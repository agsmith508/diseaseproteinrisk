import pandas as pd
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from scarf.loss import InfoNCE
from scarf.model import SCARF
from tqdm.auto import tqdm


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = get_default_device()

class SCARFDataset(Dataset):
    def __init__(self, data,  columns=None):
        self.data = np.array(data)
        self.columns = columns

    def __getitem__(self, index):
        # the dataset must return a pair of samples: the anchor and a random one from the
        # dataset that will be used to corrupt the anchor
        random_idx = np.random.randint(0, len(self))
        random_sample = torch.tensor(self.data[random_idx], dtype=torch.float)
        sample = torch.tensor(self.data[index], dtype=torch.float)

        return sample, random_sample

    def __len__(self):
        return len(self.data)

    def to_dataframe(self):
        return pd.DataFrame(self.data, columns=self.columns)

    @property
    def shape(self):
        return self.data.shape

# dataset import
x = pd.read_csv(protein_datafile, sep='\t', index_col=0)
input_size = x.shape[1]

# split training set - held out test set
x_train, x_test = train_test_split(x,  test_size=0.2, shuffle=True, random_state=0)

# scaler
scaler = StandardScaler()

# scale data
x_scaled = scaler.fit_transform(x_train)
x_scaled_test = scaler.transform(x_test)

# training params
batchsize = 128
epochs = 1000
corruption_rate = 0.7

# make model
model = SCARF(
    input_dim=input_size,
    emb_dim=256,
    hidden_dim=256,
    pretrain_dim=256,
    corruption_rate=corruption_rate,
    dropout=0.04,
    encoder_depth=4,
    head_depth=2).to(device)
    
# loss criterion
criterion = InfoNCE(temperature=1.0, negative_mode=None)

# optimizer - adam
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# datasets
train_ds = SCARFDataset(x_scaled)
test_ds = SCARFDataset(x_scaled_test)

# create data loaders
train_loader = DataLoader(train_ds, batch_size=batchsize, shuffle=True, num_workers=0, drop_last=False, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=batchsize)

# training SCARF
min_loss = 9999
stop_count = 0 
for epoch in range(epochs):
    
    rolling_loss = torch.tensor(0)
    rolling_loss_test = torch.tensor(0)
    
    for step, (anchor, random_sample) in enumerate(train_loader):

        anchor, random_sample = anchor.to(device), random_sample.to(device)
        
        # reset gradients
        optimizer.zero_grad()
        
        # get embeds
        model.train()
        emb_anchor, emb_positive = model(anchor, random_sample)
        
        # compute loss
        loss = criterion(emb_anchor, emb_positive)
        loss.backward()
        
        # update model weights
        optimizer.step()
        
        # log progress
        rolling_loss = rolling_loss.add(loss)
        
    rolling_loss = rolling_loss / (step+1)
        
    for step, (anchor, random_sample) in enumerate(test_loader):
            
        anchor, random_sample = anchor.to(device), random_sample.to(device)
        
        # get embeds
        model.eval()
        with torch.no_grad():
            emb_anchor, emb_positive = model(anchor, random_sample)
        
        # compute loss
        loss = criterion(emb_anchor, emb_positive)
        
        # log progress
        rolling_loss_test = rolling_loss_test.add(loss)
   
    rolling_loss_test = rolling_loss_test / (step+1)

    print('Epoch %i -- Loss: Train %.3f  Test %.3f' % (epoch, rolling_loss, rolling_loss_test))
