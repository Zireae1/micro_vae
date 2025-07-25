import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import Dataset, TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler
from scipy import stats
from tqdm import tqdm
# from skbio.stats.composition import clr
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

### The VAE previously used for smaller datasets
relu = torch.nn.ReLU()
#relu = torch.nn.ELU()
class VAE(torch.nn.Module):
    def __init__(self, input_dim):
        self.input_dim = input_dim
        super(VAE, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 1024)
        #self.fc2 = torch.nn.Linear(90, 80)
        self.fc3a = torch.nn.Linear(1024, 64)
        self.fc3b = torch.nn.Linear(1024, 64)
        self.fc4 = torch.nn.Linear(64, 1024)
        #self.fc5 = torch.nn.Linear(26, 52)
        self.fc6 = torch.nn.Linear(1024, input_dim)
        
        # Define proportion or neurons to dropout
        self.dropout = torch.nn.Dropout(0.1)  

    def encode(self, x):  # 784-400-[20,20]
        z = self.fc1(x)
        z = self.dropout(z)
        
        #z = torch.sigmoid(z)
        #z = torch.tanh(z)
        z = relu(z)
        z = self.dropout(z)
        
        #z = self.fc2(z)
        #z = self.dropout(z)
        
        #z = torch.sigmoid(z)
        #z = torch.tanh(z)
        #z = relu(z)
        #z = self.dropout(z)
        
        z1 = self.fc3a(z)  # u
        #z1 = self.fc3a(x)  # u
        #z1 = self.dropout(z1)
        
        z2 = self.fc3b(z)  # logvar
        #z2 = self.fc3b(x)  # logvar
        #z2 = self.dropout(z2)
        return (z1, z2)

    def decode(self, z):  # 20-400-784
        z = self.fc4(z)
        z = self.dropout(z)
        
        #z = torch.sigmoid(z)
        #z = torch.tanh(z)
        z = relu(z)
        z = self.dropout(z)
        
        #z = self.fc5(z)
        #z = self.dropout(z)
        
        #z = torch.sigmoid(z)
        #z = torch.tanh(z)
        #z = relu(z)
        #z = self.dropout(z)
        
        z = self.fc6(z)
        z = self.dropout(z)
        
        #z = torch.sigmoid(z) ### turn off for abundance prediction
        z = torch.tanh(z)
        #z = relu(z)
        return z

    def forward(self, x):  # 
        x = x.view(-1, self.input_dim)
        (u, logvar) = self.encode(x)
        stdev = torch.exp(0.5 * logvar)
        noise = torch.randn_like(stdev)
        z = u + 1 * (noise * stdev)  
        z = self.decode(z)    
        return (z, u, logvar)
    
    
    
    
## modified structure
class VAE_1(nn.Module):
    def __init__(self, input_dim, latent_dim=64, hidden_dim=1024, dropout_rate=0.1):
        super(VAE_1, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.get_mu = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim), 
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.get_logvar = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim), 
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout_rate),
            nn.Tanh() 
        )
        
    def encode(self, x):
        h = self.encoder(x)
        mu = self.get_mu(h)
        logvar = self.get_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
#     def sample(self, num_samples, device):
#         z = torch.randn(num_samples, self.fc_mu.out_features).to(device)
#         return self.decode(z)
    
    

## double decoder version:
## 1 decoder for binary output and 1 for reconstruction
class VAE_2(nn.Module):
    def __init__(self, input_dim, latent_dim=64, hidden_dim=1024, dropout_rate=0.1):
        super(VAE_2, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.get_mu = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim), 
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.get_logvar = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim), 
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.decoder_b = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        self.decoder_nb = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )
        
    def encode(self, x):
        h = self.encoder(x)
        mu = self.get_mu(h)
        logvar = self.get_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        binary = self.decoder_b(z)
        nonbinary = self.decoder_nb(z)
        return binary, nonbinary, mu, logvar
    
    

## VAE modified according to scVI
class scviVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=128, hidden_dim=1024, dropout_rate=0.1):
        super().__init__()
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        # z encoder
        self.fc_mu_z = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar_z = nn.Linear(hidden_dim, latent_dim)
        # library-size encoder (log l)
        self.fc_mu_l = nn.Linear(hidden_dim, 1)
        self.fc_logvar_l = nn.Linear(hidden_dim, 1)
        # rho decoder (mean)
        self.decoder_rho = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softplus()
        )
        # dropout decoder (pi)
        self.decoder_pi = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        # Species-specific scale factor and dispersion
        self.log_s = nn.Parameter(torch.zeros(input_dim))
        self.log_theta = nn.Parameter(torch.zeros(input_dim))

    def encode(self, x):
        h = self.encoder(x)
        mu_z = self.fc_mu_z(h)
        logvar_z = self.fc_logvar_z(h)
        mu_l = self.fc_mu_l(h)
        logvar_l = self.fc_logvar_l(h)
        return mu_z, logvar_z, mu_l, logvar_l

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, l_latent):
        rho = self.decoder_rho(z)         # positive
        pi = self.decoder_pi(z)           # [0,1]
        s = torch.exp(self.log_s)         # positive
        theta = torch.exp(self.log_theta) # positive
        l = torch.exp(l_latent) * s       # broadcast to (batch, features)
        mu = rho * l
        return mu, theta, pi

    def forward(self, x):
        # x: raw counts (batch_size, input_dim)
        mu_z, logvar_z, mu_l, logvar_l = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        l_latent = self.reparameterize(mu_l, logvar_l)
        mu, theta, pi = self.decode(z, l_latent)
        return mu, theta, pi, mu_z, logvar_z, mu_l, logvar_l