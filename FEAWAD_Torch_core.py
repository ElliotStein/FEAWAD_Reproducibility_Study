import torch

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.utils.data import Dataset

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import os
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.datasets import load_svmlight_file
from scipy.sparse import vstack
from datetime import datetime

import numpy as np
import csv

# from FEAWAD_Unchanged import inject_noise, inject_noise_sparse
from scipy.sparse import vstack, csc_matrix

def inject_noise_sparse(seed, n_out, random_seed):  
    '''
    add anomalies to training data to replicate anomaly contaminated data sets.
    we randomly swape 5% features of anomalies to avoid duplicate contaminated anomalies.
    This is for sparse data.
    '''
    rng = np.random.RandomState(random_seed) 
    n_sample, dim = seed.shape
    swap_ratio = 0.05
    n_swap_feat = int(swap_ratio * dim)
    seed = seed.tocsc()
    noise = csc_matrix((n_out, dim))
    print(noise.shape)
    for i in np.arange(n_out):
        outlier_idx = rng.choice(n_sample, 2, replace = False)
        o1 = seed[outlier_idx[0]]
        o2 = seed[outlier_idx[1]]
        swap_feats = rng.choice(dim, n_swap_feat, replace = False)
        noise[i] = o1.copy()
        noise[i, swap_feats] = o2[0, swap_feats]
    return noise.tocsr()

def inject_noise(seed, n_out, random_seed):   
    '''
    add anomalies to training data to replicate anomaly contaminated data sets.
    we randomly swape 5% features of anomalies to avoid duplicate contaminated anomalies.
    this is for dense data
    ''' 
    rng = np.random.RandomState(random_seed) 
    n_sample, dim = seed.shape
    swap_ratio = 0.05
    n_swap_feat = int(swap_ratio * dim)
    noise = np.empty((n_out, dim))
    for i in np.arange(n_out):
        outlier_idx = rng.choice(n_sample, 2, replace = False)
        o1 = seed[outlier_idx[0]]
        o2 = seed[outlier_idx[1]]
        swap_feats = rng.choice(dim, n_swap_feat, replace = False)
        noise[i] = o1.copy()
        noise[i, swap_feats] = o2[swap_feats]
    return noise

# Functions

def flatten(x):
    return x.view(784)

def filter_dataset(dataset, target_class, include=True):
    """
    Returns datasubset object of dataset. 
    If include = True, only data with target = target_class (anomalies)
    else: only data without target = target_class (normal)
    """
    targets = dataset.targets
    idx = np.arange(0, len(dataset))

    target_class = 0
    if include:
        idx_new = targets[idx]==target_class
    else:
        idx_new = targets[idx]!=target_class

    # Only keep your desired classes
    idx_new = idx[idx_new]

    return Subset(dataset, idx_new)

def test_dataset_purity(dataset, anomaly_class):
    anom = dataset[0][1] == anomaly_class
    for i in range(len(dataset)):
        if (dataset[i][1] == anomaly_class) != anom:
            print("Not all data are anomalous/normal")
            print("Found digit {} at index {}".format(dataset[i][1], i))
            return

    print("Dataset contains {} only".format("Anomalies" if anom else "normal data"))

def get_ckpt_path(checkpoint_dir, version_num=-1):
    version = os.listdir(checkpoint_dir +"/lightning_logs")[version_num]
    ckpt = os.listdir(checkpoint_dir +"/lightning_logs/" + version +"/checkpoints")[-1]
    ckpt_path = checkpoint_dir +"/lightning_logs/" + version + "/checkpoints/" + ckpt
    return ckpt_path

def aucPerformance(mse, labels):

    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    return roc_auc, ap;

def write_details(results_dir=os.getcwd(), initial_write=False, **kwargs):
    # Create details.txt file to store experiment details

    # res = experiment_folder(results_dir)
    with open(results_dir + 'Details.txt', 'a') as the_file:
        if initial_write:
                the_file.write("Experimental details: ")
        the_file.write('\n\n\n')
        for arg,value in kwargs.items():
            the_file.write(arg + '=' + str(value) + '\n')
    
    return results_dir

def make_checkpoint_dir(results_dir):
    # results_dir = experiment_folder(results_dir)
    if "CheckpointsAE" not in os.listdir(results_dir):
        os.mkdir(results_dir + "/CheckpointsAE")
    if "CheckpointsAS" not in os.listdir(results_dir):
        os.mkdir(results_dir + "/CheckpointsAS")
    return results_dir+"CheckpointsAE", results_dir+"CheckpointsAS"

def experiment_folder(res):
    pth = res + "/" + str(datetime.now()).split()[0] + " || "+ str(datetime.now()).split()[1].split(":")[0] + " | " + str(datetime.now()).split()[1].split(":")[1]  + "/"
    if os.path.isdir(pth) == True:
        print("Results directory exists, creating temporary sub directory")
        pth = pth + str(datetime.now()).split('.')[-1] + "/"
    os.mkdir(pth)
    print('Make Experiment Directory')
    return pth

# Functions for .csv data loading from original FEAWAD and toolsdev files, from FEAWAD repo
# These have been packaged into a single function csv_data_setup, which returns everything Torch needs to build a dataset class

def get_data_dim(dataset, data_type):
    """
    Takes dataset Str and data_type (currently only datatype 0 is accepted)
    returns length of first item in dataset
    """
    if '_normalization' not in dataset:
        dataset += '_normalization'
    if "spambase" in dataset:
        labels_dim=1
    else:
        labels_dim=2
    
    if data_type == 0 or '0':
        if '.csv' not in dataset:
            dataset += '.csv'
    if './dataset/' not in dataset:
        dataset = './dataset/' + dataset
    
    try:
        with open(dataset, 'r') as file:
            print('Dataset found')
            reader = csv.reader(file)
            return len(next(iter(reader))) - labels_dim
    except:
        raise ValueError("File not found: {}".format(dataset))

def csv_data_setup(dataset_name, random_seed, data_format, input_path, cont_rate, known_outliers): 
    """
    This takes care of the data generation steps in the original FEAWAD implementation.
    It returns numpy arrays of data and labels, in a format ready for a custom Torch dataset class to use
    """


    nm = dataset_name
    
    filename = nm.strip()

    data_dim = get_data_dim(nm, 0) #### custom, was previously an argument
    if data_format == 0 or data_format == '0':
        x, labels = dataLoading(input_path + filename + ".csv", byte_num=data_dim) ####
    else:
        x, labels = get_data_from_svmlight_file(input_path + filename + ".svm")
        x = x.tocsr()    
    outlier_indices = np.where(labels == 1)[0]
    outliers = x[outlier_indices]
    n_outliers_org = outliers.shape[0]

    print("dataLoading input:", input_path + filename + ".csv", data_dim)
    # print("train_test_split input labels:", labels)
    train_x, test_x, train_label, test_label = train_test_split(x, labels, test_size=0.2, random_state=42, stratify = labels)

    # print(filename + ': round ' + str(i))
    outlier_indices = np.where(train_label == 1)[0]
    inlier_indices = np.where(train_label == 0)[0]
    n_outliers = len(outlier_indices)
    print("Original training size: %d, Number of outliers in Train data:: %d" % (train_x.shape[0], n_outliers))
    
    n_noise  = len(np.where(train_label == 0)[0]) * cont_rate / (1. - cont_rate)
    n_noise = int(n_noise)    
    
    rng = np.random.RandomState(random_seed)  
    if data_format == 0 or data_format == '0':                
        if n_outliers > known_outliers:
            mn = n_outliers - known_outliers
            remove_idx = rng.choice(outlier_indices, mn, replace=False)
            train_x = np.delete(train_x, remove_idx, axis=0)
            train_label = np.delete(train_label, remove_idx, axis=0)
            #ae_label = train_x
        noises = inject_noise(outliers, n_noise, random_seed)
        train_x = np.append(train_x, noises, axis = 0)
        train_label = np.append(train_label, np.zeros((noises.shape[0], 1)))
    
    else: # Only format 0 is supported currently
        if n_outliers > known_outliers:
            mn = n_outliers - known_outliers
            remove_idx = rng.choice(outlier_indices, mn, replace=False)        
            retain_idx = set(np.arange(train_x.shape[0])) - set(remove_idx)
            retain_idx = list(retain_idx)
            train_x = train_x[retain_idx]
            train_label = train_label[retain_idx]                               
        
        noises = inject_noise_sparse(outliers, n_noise, random_seed)
        train_x = vstack([train_x, noises])
        train_label = np.append(train_label, np.zeros((noises.shape[0], 1)))
    
    outlier_indices = np.where(train_label == 1)[0]
    inlier_indices = np.where(train_label == 0)[0]

    return train_x, inlier_indices, outlier_indices, data_dim, test_x, test_label

# From toolsdev.py
def get_data_from_svmlight_file(path):
    data = load_svmlight_file(path)
    return data[0], data[1]

def dataLoading(path, byte_num):
    # loading data
    x=[]
    labels=[]
    
    with (open(path,'r')) as data_from:
        csv_reader=csv.reader(data_from)
        for i in csv_reader:
            x.append(i[0:byte_num])
            labels.append(i[byte_num])

    for i in range(len(x)):
        for j in range(byte_num):
            x[i][j] = float(x[i][j])
    for i in range(len(labels)):
        labels[i] = float(labels[i])
    x = np.array(x)
    labels = np.array(labels)

    return x, labels;

def aucPerformance(mse, labels):

    try:
        roc_auc = roc_auc_score(labels, mse)
    except:
        print('labels = ',labels, 'preds =', mse)
        raise ValueError("roc_auc fails")
    ap = average_precision_score(labels, mse)
    # print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    return roc_auc, ap;

def writeResults(name, n_samples_trn,  n_outliers, n_samples_test,test_outliers ,test_inliers, avg_AUC_ROC, avg_AUC_PR, std_AUC_ROC,std_AUC_PR, path):    
    csv_file = open(path, 'a') 
    row = name + ","  + n_samples_trn + ','+n_outliers  + ','+n_samples_test+','+test_outliers+','+test_inliers+','+avg_AUC_ROC+','+avg_AUC_PR+','+std_AUC_ROC+','+std_AUC_PR + "\n"
    csv_file.write(row)

# Classes

class ae_unlabeled(Dataset):
    def __init__(self, dataset_norm, dataset_anom, contaminant_prob, transform=None, target_transform=None):
        self.dataset_norm = dataset_norm
        self.dataset_anom = dataset_anom
       
        self.contaminant_prob = contaminant_prob


    def __len__(self):
        return len(self.dataset_norm)
    
    def __getitem__(self, idx):
        if self.with_prob():
            idx = np.random.randint(len(self.dataset_anom))
            data = self.dataset_anom[idx][0]
        else:
            data = self.dataset_norm[idx][0]

        # Autoencoders require target = input
        return data, data
        

    def with_prob(self):
        if np.random.rand() < self.contaminant_prob:
            return True
        else:
            return False

class anomaly_score(Dataset):
    def __init__(self, dataset_norm, dataset_anom, known_anoms, contaminant_prob, transform=None, target_transform=None):
        self.dataset_norm = dataset_norm
        self.dataset_anom = dataset_anom
        # Create list of k indices corresponding to elements in dataset_anom
        self.eval = True
        if known_anoms != 0:
            self.eval = False
            self.anom_indices = np.random.choice(np.arange(len(dataset_anom)), known_anoms, replace=False)
            self.anom_pseudodataset = np.random.choice(self.anom_indices, len(dataset_norm), replace=True)
        else:
            self.anom_indices, self.anom_pseudodataset = [],[]

        self.contaminant_prob = contaminant_prob

    def __len__(self):
        if self.eval:
            return len(self.dataset_norm)
        else:
            return len(self.dataset_norm)*2
    
    def __getitem__(self, idx):
        # sample alternately from unlabeled and labeled_anomaly set
        # unlabeled is a mix of normal data and a small probability of incorrectly labeled anomaly data
        # labeled anomalies are found by first fixing a subset of known_anoms. Then creating an oversampled dataset by choosing len(dataset_norm) items from this list (with replacement)
        if self.eval:
            return self.get_item_eval(idx)
        
        if idx//2 == idx/2:
            if self.with_prob():
                idx = self.get_contaminant()
                data = self.dataset_anom[idx][0]
                # print("Contaminant: {}".format(idx))
            else:
                data = self.dataset_norm[idx//2][0]
                # print("norm: {}".format(idx))
            return data, 0.0
        else:
            # print("anom: {}".format(self.anom_pseudodataset[idx]))
            return self.dataset_anom[self.anom_pseudodataset[idx//2]][0], 1.0

    def get_item_eval(self, idx):
        if self.with_prob():
            idx = self.get_contaminant()
            data = self.dataset_anom[idx][0]
            label = 1.0
            # print("Contaminant: {}".format(idx))
        else:
            data = self.dataset_norm[idx][0]
            label = 0.0
            # print("norm: {}".format(idx))
        return data, label

    def with_prob(self):
        if np.random.rand() < self.contaminant_prob:
            return True
        else:
            return False
    
    def get_contaminant(self):
        idx = np.random.randint(len(self.dataset_anom))
        while idx in self.anom_indices:
            idx = np.random.randint(len(self.dataset_anom))

        return idx

class AutoEncoder(pl.LightningModule):
    def __init__(self, input_len):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_len, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 50))
        self.decoder = nn.Sequential(
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 300),
            nn.ReLU(),
            nn.Linear(300, 400),
            nn.ReLU(),
            nn.Linear(400, input_len),
        )
        self.preds, self.labels = np.array([]), np.array([])
        
    def forward(self, data):
        data = data.view(data.size(0), -1)
        embedding = self.encoder(data)
        recon = self.decoder(embedding)
        return recon

    def get_embedding(self, x):
        return self.encoder(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        data, _ = train_batch
        recon = self.forward(data)
        loss = F.mse_loss(recon.view(recon.size(0), -1), data.view(data.size(0), -1))
        self.log('ae_train_loss', loss)
        return loss

    def anomaly_prediction(self, batch):
        recon = self.forward(batch)
        square_error = (batch - recon)**2
        mse = torch.mean(square_error, dim=list(range(1, len(square_error.shape))))
        return mse

    def validation_step(self, eval_batch, eval_batch_idx):
        data, label = eval_batch

        mse = self.anomaly_prediction(data).detach().cpu()
        
        self.preds  = np.append(self.preds, mse)
        self.labels = np.append(self.labels, label.detach().cpu().numpy())

    def get_aucs(self, log=True, return_current_vals=False):
        if return_current_vals: # Does not interfere with val procedure, can be used seperately
            return  self.auc_roc, self.auc_pr
        
        # Must be called after eval epoch to reset preds and labels
        self.auc_roc, self.auc_pr = aucPerformance(mse=self.preds, labels=self.labels)
        
        if log:
            self.log("AUC_ROC", self.auc_roc)
            self.log("AUC_pr", self.auc_pr)

        self.preds = np.array([])
        self.labels = np.array([])

class AnomalyScoreModel(pl.LightningModule): 
    def __init__(self, input_len, ae_hidden, ckpt_path, AEmodel_class):
        super().__init__()
        
        self.AEmodel = AEmodel_class.load_from_checkpoint(ckpt_path, input_len=input_len)

        self.dense1 = nn.Linear(input_len+ae_hidden+1, 256)
        self.dense2 = nn.Linear(256 + 1, 32)
        self.dense3 = nn.Linear(32  + 1, 1)

        self.relu = nn.ReLU()

        self.preds = np.array([])
        self.labels = np.array([])

        self.auc_roc, self.auc_pr = 0,0

    def get_encoding_and_recon(self, x):
        x = x.view(x.size(0), -1)
        enc = self.AEmodel.encoder(x)
        rec = self.AEmodel.decoder(enc)
        return enc, rec

    def forward(self, data):
        # running through AE is handled with seperate function for easy modification by inherited classes
        enc, rec = self.get_encoding_and_recon(data)
        residual = data - rec
        residual = residual.view(residual.size(0), -1)

        # print(residual.shape)
        if len(enc.shape) == 1:
            # Single item, not batch
            recon_error = torch.linalg.norm(residual).unsqueeze(0)
        else:
            recon_error = torch.linalg.norm(residual, dim=1).unsqueeze(1)
        
        # print(recon_error.shape)
        # print("residual:{} | recon_error: {}".format(residual.shape, recon_error.shape))
        residual_normalised = residual/recon_error #torch.div(residual, recon_error)
        # print("residual_normalised:{}, enc:{}, recon:{}".format(residual_normalised.shape, enc.shape, recon_error.shape))

        combined_input = torch.cat((residual_normalised.T, enc.T, recon_error.T)).T

        out = self.dense1(combined_input)
        out = self.relu(out) 
        # print("out:{} recon_error:{} cat:{}".format(out.shape, recon_error.shape, torch.cat((out, recon_error), dim=1).shape))
        if len(data.shape) == 1:
            out = torch.cat((out, recon_error), dim=0)
        else:
            out = torch.cat((out, recon_error), dim=1)
        # print("cat:{}".format(out.shape))

        out = self.dense2(out)
        out = self.relu(out)
        
        if len(data.shape) == 1:
            out = self.dense3(torch.cat((out, recon_error), dim=0))
        else:
            out = self.dense3(torch.cat((out, recon_error), dim=1))
        return out, recon_error

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        data, label = train_batch

        pred, recon_error = self.forward(data)
        loss = self.loss(pred, label, recon_error)

        self.log("ae_recon_loss", recon_error.mean())
        self.log('as_train_loss', loss)
        return loss
    
    def validation_step(self, eval_batch, eval_batch_idx):
        data, label = eval_batch

        preds = self.forward(data)[0].detach().cpu().numpy()

        self.preds  = np.append(self.preds, preds)
        self.labels = np.append(self.labels, label.detach().cpu().numpy())

    def get_aucs(self, log=True, return_current_vals=False):
        if return_current_vals: # Does not interfere with val procedure, can be used seperately
            return  self.auc_roc, self.auc_pr
        
        # Must be called after eval epoch to reset preds and labels
        self.auc_roc, self.auc_pr = aucPerformance(self.preds, self.labels)
        
        if log:
            self.log("AUC_ROC", self.auc_roc)
            self.log("AUC_pr", self.auc_pr)

        self.preds = np.array([])
        self.labels = np.array([])
        
    def loss(self, pred, true, recon_error):
        ae_loss = (1-true)*recon_error + true * torch.max(torch.zeros_like(recon_error), 5 - recon_error)
        ae_loss = ae_loss.sum()

        anomaly_score_loss = (1-true)*torch.abs(pred) + true * torch.max(torch.zeros_like(pred), 5 - pred)
        anomaly_score_loss = anomaly_score_loss.sum()

        return ae_loss + anomaly_score_loss

class auc(Callback):

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the val epoch ends."""
        pl_module.get_aucs(log=True)

#

class ae_unlabelled_csv_data(Dataset):
    def __init__(self, train_x, inlier_indices):
        super(Dataset).__init__()
        
        random_seed = 42
        self.rng = np.random.RandomState(random_seed)
        self.train_x = train_x
        self.inlier_indices = inlier_indices
        self.len = len(self.inlier_indices)
        self.random_indices = np.random.permutation(self.len)

    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):

        rand_inx = self.random_indices[index]
        data = self.train_x[self.inlier_indices[rand_inx]] # choose the sid'th inlier index. Then get the training data corresponding
        
        return torch.Tensor(data), torch.Tensor(data) # For autoencoders

class anomaly_score_csv_data(Dataset):
    def __init__(self, train_x, outlier_indices, inlier_indices):
        super(Dataset).__init__()
        
        random_seed = 42
        self.rng = np.random.RandomState(random_seed)
        self.train_x = train_x
        self.inlier_indices = inlier_indices
        self.outlier_indices = outlier_indices

        self.random_indices_inliers = np.random.permutation(len(self.inlier_indices))
        self.random_indices_outliers = np.random.choice(outlier_indices, len(inlier_indices), replace=True)

    def __len__(self):
        return len(self.random_indices_inliers) + len(self.random_indices_outliers)
    
    def __getitem__(self, index):

        if(index % 2 == 0):   
            
            data = self.train_x[self.random_indices_inliers[index//2]]
            training_label = [0.0]
        else:
            
            data = self.train_x[self.random_indices_outliers[index//2]]
            training_label = [1.0]
        
        return torch.Tensor(data), torch.Tensor(training_label)

class anomaly_score_csv_data_eval(Dataset):

    def __init__(self, test_x, test_labels):
        super(Dataset).__init__()
        
       
        self.test_x = test_x
        self.test_labels = test_labels
    
    def __len__(self):
        return len(self.test_x)
    
    def __getitem__(self, index):

        data = self.test_x[index]
        label = self.test_labels[index]
        
        return torch.Tensor(data), torch.Tensor([label])

