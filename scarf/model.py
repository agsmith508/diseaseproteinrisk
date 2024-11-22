import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class L2NormalizationLayer(nn.Module):
    def __init__(self, dim=1, eps=1e-6):
        super(L2NormalizationLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)

class MLP(torch.nn.Sequential):
    """Simple multi-layer perceptron with ReLu activation and optional dropout layer"""

    def __init__(self, input_dim, hidden_dim, embed_dim, n_layers, dropout=0.04):
        layers = []
        in_dim = input_dim
        for _ in range(n_layers - 1):
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, embed_dim))

        super().__init__(*layers)


        
class L2NormMLP(torch.nn.Sequential):
    """Simple multi-layer perceptron with ReLu activation and optional dropout layer"""

    def __init__(self, input_dim, hidden_dim, embed_dim, n_layers, dropout=0.04):
        layers = []
        in_dim = input_dim
        for _ in range(n_layers - 1):
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(L2NormalizationLayer())
            layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, embed_dim))

        super().__init__(*layers)

class SCARF(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim,
        hidden_dim,
        pretrain_dim,
        encoder_depth=4,
        head_depth=2,
        corruption_rate=0.6,
        encoder=None,
        pretraining_head=None,
        dropout=0.04,
        pretraining=False
    ):
        """Implementation of SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption.
        It consists in an encoder that learns the embeddings.
        It is done by minimizing the contrastive loss of a sample and a corrupted view of it.
        The corrupted view is built by remplacing a random set of features by another sample randomly drawn independently.

            Args:
                input_dim (int): size of the inputs
                emb_dim (int): dimension of the embedding space
                encoder_depth (int, optional): number of layers of the encoder MLP. Defaults to 4.
                head_depth (int, optional): number of layers of the pretraining head. Defaults to 2.
                corruption_rate (float, optional): fraction of features to corrupt. Defaults to 0.6.
                encoder (nn.Module, optional): encoder network to build the embeddings. Defaults to None.
                pretraining_head (nn.Module, optional): pretraining head for the training procedure. Defaults to None.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim=emb_dim
        self.hidden_dim=hidden_dim
        self.pretrain_dim=pretrain_dim
        self.encoder_depth = encoder_depth
        self.head_depth = head_depth
        self.corruption_rate = corruption_rate
        self.dropout = dropout
        self.emb_dim=emb_dim
        self.pretraining = pretraining

        if encoder:
            self.encoder = encoder
        else:
            self.encoder = MLP(self.input_dim, self.hidden_dim, self.emb_dim, self.encoder_depth, self.dropout)

        if pretraining_head:
            self.pretraining_head = pretraining_head
        else:
            self.pretraining_head = L2NormMLP(self.emb_dim, self.pretrain_dim, self.emb_dim, self.head_depth, self.dropout)
            
        self.encoder.apply(self._init_weights)
        self.pretraining_head.apply(self._init_weights)
        self.corruption_len = int(corruption_rate * input_dim)

        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def forward(self, anchor, random_sample):
        batch_size, m = anchor.size()
        
        corruption_mask = torch.zeros_like(anchor, dtype=torch.bool)
        for i in range(batch_size):
            corruption_idx = torch.randperm(m)[: self.corruption_len]
            corruption_mask[i, corruption_idx] = True
            
        positive = torch.where(corruption_mask, random_sample, anchor)

        # compute embeddings
        emb_anchor = self.encoder(anchor)
        emb_anchor = self.pretraining_head(emb_anchor)

        emb_positive = self.encoder(positive)
        emb_positive = self.pretraining_head(emb_positive)

        return emb_anchor, emb_positive


    def get_embeddings(self, input):
        return self.encoder(input)

class MLPCL(torch.nn.Sequential):
    """Simple multi-layer perceptron with ReLu activation and optional dropout layer"""

    def __init__(self, input_dim, hidden_dim, n_layers, dropout, n_classes=2):
        layers = []
        in_dim = input_dim
        for _ in range(n_layers - 1):
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU())
            layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, n_classes))

        super().__init__(*layers)

class SCARFCLS(nn.Module):
    def __init__(
        self,
        pretrained_model,
        dropout,
        input_dim,
        hidden_dim,
        n_layers,
    ):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.cls_mlp = MLPCL(input_dim, hidden_dim, n_layers, dropout=dropout)
        self.cls_mlp.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def forward(self, x):
        encoded = self.pretrained_model.get_embeddings(x)
        logits = self.cls_mlp(encoded)
        return logits
    
        
        