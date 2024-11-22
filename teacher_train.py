import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from scarf.model import SCARF, SCARFCLS
from sklearn.metrics import roc_auc_score, brier_score_loss
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = get_default_device()

class StratifiedBatchSampler:
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, y, batch_size, shuffle=True):
        if torch.is_tensor(y):
            y = y.numpy()
        assert len(y.shape) == 1, 'label array must be 1D'
        n_batches = int(len(y) / batch_size)
        self.skf = StratifiedKFold(n_splits=n_batches, shuffle=shuffle)
        self.X = torch.randn(len(y),1).numpy()
        self.y = y
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = torch.randint(0,int(1e8),size=()).item()
        for train_idx, test_idx in self.skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        return len(self.y)

    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def mixup_data(Xs: torch.Tensor, Ys: torch.Tensor, alpha=0.4):
    b, f = Xs.shape
    lam = torch.distributions.beta.Beta(alpha, alpha).sample().item()
    shuffle_sample_ids = torch.randperm(b)
    
    mixed_X = lam * Xs + (1 - lam) * Xs[shuffle_sample_ids, :]
    mixed_Y = lam * Ys + (1 - lam) * Ys[shuffle_sample_ids]
    return mixed_X.to(torch.float32), mixed_Y.to(torch.float32)

# training params
epochs = 1000
condition = 1 # index for condition of interest

params = pd.read_pickle('model_dict.pkl')
params = params[condition]

# unpack params
init_seed = params['init_seed']
lr = params['lr']
weight_decay = params['weight_decay']
grad_norm = params['grad_norm']
batch_size = int(params['batch_size'])
n_layers = params['n_layers']
hidden_size = params['hidden_size']
weighted_loss = params['weighted_loss']
mixup = params['mixup']
smote = params['smote']
dropout_rate = params['dropout_rate']
mixup_rate = params['mixup_rate']
smote_strat = params['smote_strat']
smoothing_rate = params['smoothing_rate']

# set seed
torch.manual_seed(init_seed)
np.random.seed(init_seed)

# dataset - olink
x = pd.read_csv(protein_datafile, sep='\t', index_col=0)
input_size = x.shape[1]

# disease predictions
y = pd.read_csv(prev_disease_datafile, sep='\t', index_col=0)

# split training set - held out test set
x_train, x_test, y_train, y_test = train_test_split(x, y,  test_size=0.2, shuffle=True, stratify=y, random_state=0)

# scaler
scaler = StandardScaler()

# scale data
x_scaled = scaler.fit_transform(x_train)
x_scaled_test = scaler.transform(x_test)

# over sampling to address class imbalance
if smote:
	oversample = SMOTE(sampling_strategy=smote_strat)
	x_scaled, y_train = oversample.fit_resample(x_scaled, y_train)

# counting different class sizes
pos = y_train.value_counts()[1]
neg = y_train.value_counts()[0]
total = pos + neg

# load pretrained scarf model
model = SCARF(
	input_dim=input_size,
	emb_dim=256,
	hidden_dim=256,
	pretrain_dim=256,
	corruption_rate=0.7,
	dropout=0.04,
	encoder_depth=4,
	head_depth=2).to(device)
model.load_state_dict(torch.load('scarf_model', map_location=torch.device('cpu')))

# create classification head 
cls_model = SCARFCLS(
		pretrained_model=model,
		dropout=dropout_rate,
		input_dim=256,
		hidden_dim=hidden_size,
		n_layers=n_layers).to(device)

# loss criterion
if weighted_loss:
	weight_for_0 = (1 / neg) * (total / 2.0)
	weight_for_1 = (1 / pos) * (total / 2.0)
	criterion = nn.CrossEntropyLoss(label_smoothing=smoothing_rate, weight=torch.tensor([weight_for_0, weight_for_1], device=device))
else:
	criterion = nn.CrossEntropyLoss(label_smoothing=smoothing_rate)

# optimizer - adam
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# make datasets
train_ds = TensorDataset(torch.tensor(x_scaled), torch.tensor(y_train))
test_ds = TensorDataset(torch.tensor(x_scaled_test), torch.tensor(y_test))

# create data loaders
train_loader = DataLoader(train_ds, pin_memory=True, batch_sampler=StratifiedBatchSampler(y_train, batch_size))
test_loader = DataLoader(test_ds, batch_size=batch_size)

# training teacher
min_loss = 999
stop_count = 0 
for epoch in range(epochs):
	
	rolling_loss = torch.tensor(0)
	rolling_loss_test = torch.tensor(0)

	one_hots = np.empty(shape=(0, 2))
	y_scores = np.empty(shape=(0, 2))
	for step, (inputs, labels) in enumerate(train_loader):

		inputs, labels = inputs.to(device).to(torch.float32), labels.to(device)
		labels = F.one_hot(labels.to(torch.int64), num_classes=2).to(torch.float32)
		one_hots = np.concatenate((one_hots, labels.cpu().numpy()), axis=0)
		
		if mixup:
			inputs, labels = mixup_data(inputs, labels)
		
		# reset gradients
		optimizer.zero_grad()
		
		# get predictions
		model.train()
		cls_model.train()
		outputs = cls_model(inputs)
		
		# compute loss
		loss = criterion(outputs, labels)
		loss.backward()
		
		# grad clipping
		if grad_norm > 0:
			nn.utils.clip_grad_norm_(cls_model.parameters(), grad_norm)
		
		# update model weights
		optimizer.step()
		
		# log progress
		rolling_loss = rolling_loss.add(loss)
	
		# log predicted scores
		y_scores = np.concatenate((y_scores, outputs.detach().cpu().numpy()), axis=0)

	# calculate performance
	rolling_loss = rolling_loss / (step+1)
	auc = roc_auc_score(one_hots, y_scores)
	brier = brier_score_loss(one_hots.argmax(1), F.softmax(torch.tensor(y_scores), dim=1)[:, 1])

	# test set
	one_hots = np.empty(shape=(0, 2))
	y_scores = np.empty(shape=(0, 2))
	for step, (inputs, labels) in enumerate(test_loader):
			
		inputs, labels = inputs.to(device).to(torch.float32), labels.to(device)
		labels = F.one_hot(labels.to(torch.int64), num_classes=2).to(torch.float32)
		one_hots = np.concatenate((one_hots, labels.cpu().numpy()), axis=0)

		# get predictions
		model.eval()
		cls_model.eval()
		with torch.no_grad():
			outputs = cls_model(inputs)
		
		# compute loss
		loss = criterion(outputs, labels)
		
		# log progress
		rolling_loss_test = rolling_loss_test.add(loss)
		
		# log predicted scores
		y_scores = np.concatenate((y_scores, outputs.detach().cpu().numpy()), axis=0)
   
	# calculate performance
	rolling_loss_test = rolling_loss_test / (step+1)
	auc_test = roc_auc_score(one_hots, y_scores)
	brier_test = brier_score_loss(one_hots.argmax(1), F.softmax(torch.tensor(y_scores), dim=1)[:, 1])

	print('-'*25)
	print('Epoch %i -- Train: Loss %.3f  AUC %.3f  Brier %.3f' % (epoch, rolling_loss, auc, brier))
	print('Epoch %i -- Test:   Loss %.3f  AUC %.3f  Brier %.3f' % (epoch, rolling_loss_test, auc_test, brier_test))

    
