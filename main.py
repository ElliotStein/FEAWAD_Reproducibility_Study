from email.policy import default
from gc import callbacks
import FEAWAD_Torch_core
import os
from torchvision.datasets import DatasetFolder
from torch.utils.data import Dataset
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import imageio
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from FEAWAD_Torch_core import *
import argparse
from time import time

def anomaly_target(target, anomaly_class=0):
        """
        Sets targets of data belonging to anomaly class to 1, and all others to zero.
        To be used as target transform in Dataset class
        """
        if target == anomaly_class:
            return 1.0
        else:
            return 0.0

class Dataset(DatasetFolder):
    def __init__(self, root , classes, transform,loader = imageio.imread, extensions=".png", target_transform=None):

        self.classes = classes
        super().__init__(root, loader, extensions, transform, target_transform)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folders in a dataset.

        See :class:`DatasetFolder` for details.
        """
        
        return self.classes, {str(i): i for i in self.classes}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        data = torch.Tensor(self.loader(path))[:,:,0].unsqueeze(0)
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

class AutoEncoderSPMNIST(AutoEncoder):
    def __init__(self, input_len):
        super().__init__(input_len) #Required for super init but not used
        self.convEncoder = nn.Sequential(
            nn.Conv2d(1, 8, 6, stride = 1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, stride = 2),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, stride = 2),
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )
        self.denseEncoder = nn.Sequential(
            nn.Linear(8*6*6, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 50)
            )

        ###

        self.denseDecoder = nn.Sequential(
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 8*6*6),
            nn.ReLU()
            )

        self.convDecoder = nn.Sequential(
            nn.ConvTranspose2d(8,8,3,stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(8,8,3,stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.ConvTranspose2d(8, 1, 6,stride=1),
            nn.ReLU(),
        )
        
    def forward(self, x):
        x = self.convEncoder(x)
        x = x.view(x.size(0), -1)
        x = self.denseEncoder(x)
        x = self.denseDecoder(x)
        x = x.view(x.size(0), 8, 6, 6)
        x = self.convDecoder(x)
        return x

class AnomalyScoreModelSPMNIST(AnomalyScoreModel):
    def __init__(self,  input_len, ae_hidden, ckpt_path, AEmodel_class):
        super().__init__(input_len, ae_hidden, ckpt_path, AEmodel_class)
        
    def get_encoding_and_recon(self, x):
        x = self.AEmodel.convEncoder(x)
        x = x.view(x.size(0), -1)
        enc = self.AEmodel.denseEncoder(x)
        x = self.AEmodel.denseDecoder(enc)
        x = x.view(x.size(0), 8, 6, 6)
        recon = self.AEmodel.convDecoder(x)
        return enc, recon

class AutoEncoderCSV(AutoEncoder):
    def __init__(self, input_len):
        super().__init__(input_len)
        self.encoder = nn.Sequential(
            nn.Linear(input_len, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_len),
            nn.ReLU()
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

class AnomalyScoreModelCSV(AnomalyScoreModel):
    def __init__(self, input_len, ae_hidden, ckpt_path, AEmodel_class):
        super().__init__(input_len, ae_hidden, ckpt_path, AEmodel_class)
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        flatten
    ])

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default = 128, help = "batch size used in SGD")
parser.add_argument("--ASepochs", type=int, default = 50, help="the number of epochs")
parser.add_argument("--AEepochs", type=int, default = 50, help="the number of epochs")
parser.add_argument("--runs", type=int, default = 10, help="how many times we repeat the experiments to obtain the average performance")
parser.add_argument("--known_outliers", type=int, default = 30, help="Outliers labelled and used in training")
parser.add_argument("--contamination_rate", type=float, default=0.02)
parser.add_argument("--IRIDIS", type=bool,default=False,help="If true, use IRIDIS path to source folder, if false, use local")
parser.add_argument("--dataset", type=str,default="SPMNIST",help="MNIST or SPMNIST")
parser.add_argument("--notes", type=str,default="",help="Notes printed in Details.txt")

csv_datasets = []
for name in os.listdir(os.getcwd() + "/dataset"):
    if 'joblib' not in name:
        csv_datasets.append(name.split('.')[0])

if __name__ == "__main__":

    args = parser.parse_args()
    time0 = time()

    if args.IRIDIS:
        root = '/mainfs/scratch/es3e20/data/'
    else:
        root = '/Users/elliotstein/Documents/PhD Year 1/Data/'
    train_root = root + 'Spoken_MNIST/training-spectrograms'
    test_root = root + 'Spoken_MNIST/testing-spectrograms'
    mnist_root = root
       
    # args written explicitly for ease of changing during development

    AEepochs=args.AEepochs
    ASepochs=args.ASepochs
    known_outliers = args.known_outliers
    dataset_name=args.dataset
    runs=args.runs
    batch_size=args.batch_size
    contamination_rate = args.contamination_rate

    num_workers = 6
    gpus=torch.cuda.device_count()

    res_dir = experiment_folder(os.getcwd())
    write_details(res_dir, initial_write=True, AEepochs=AEepochs, ASepochs=ASepochs, runs=runs, known_outliers=known_outliers, dataset_name = dataset_name, 
        notes=args.notes)
    checkpoint_ae_dir, checkpoint_as_dir=make_checkpoint_dir(res_dir)

    auc_rocs, auc_prs = [],[]

    """
    Whichever dataset we use, we need to set up:
        dataset_ae_train - unlabelled data with anomaly samples at contamination_rate. Labels unused
        dataset_as_train - 50% same as above, other 50% oversampled from the labeled anomaly set. Labels 1 for anom and 0 for normal.
        dataset_eval - Full eval partition dataset (all classes, anomaly and normal). Labels 1 for anom and 0 for normal.

        ae_hidden - dimension of AE bottleneck
        ae_model_class - Un-initialised class object to be used for autoencoder
        as_model_class - Un-initialised class object to be used for anomaly score model
    
    """
    if dataset_name == "SPMNIST":
        dataset_norm_train = Dataset(root=train_root, classes=[1,2,3,4,5,6,7,8,9], transform=transforms.Normalize(120.5, 40.78))
        dataset_anom_train = Dataset(root=train_root, classes=[0], transform=transforms.Normalize(120.5, 40.78))

        dataset_eval = Dataset(root=test_root, classes=[0,1,2,3,4,5,6,7,8,9], transform=transforms.Normalize(120.5, 40.78), target_transform=anomaly_target)

        ae_hidden=50
        ae_model_class = AutoEncoderSPMNIST
        anomaly_score_model_class = AnomalyScoreModelSPMNIST

    elif dataset_name == "MNIST":
        dataset_train = MNIST(root=mnist_root, download=False, train=True, transform=mnist_transform)
        dataset_eval = MNIST(root=mnist_root, download=False, train=False, transform=mnist_transform, target_transform=anomaly_target)
        
        dataset_norm_train = filter_dataset(dataset_train, 0, include=False)
        dataset_anom_train = filter_dataset(dataset_train, 0, include=True)

        ae_hidden=50
        ae_model_class = AutoEncoder
        anomaly_score_model_class = AnomalyScoreModel

    elif dataset_name in csv_datasets:
        train_x, inlier_indices, outlier_indices, data_dim, test_x, test_label = csv_data_setup(dataset_name, 42, 0, './dataset/', contamination_rate, known_outliers)

        dataset_ae_train = ae_unlabelled_csv_data(train_x, inlier_indices)
        dataset_as_train = anomaly_score_csv_data(train_x, outlier_indices, inlier_indices)
        dataset_eval = anomaly_score_csv_data_eval(test_x, test_label)

        ae_hidden=64
        ae_model_class = AutoEncoderCSV
        anomaly_score_model_class = AnomalyScoreModelCSV
    else:
        raise ValueError("Dataset not recognised. Must be one of:{}{}".format("SPMIST, MNIST,", csv_datasets))

    if dataset_name not in csv_datasets:
        dataset_ae_train = ae_unlabeled(dataset_norm_train, dataset_anom_train, contamination_rate)
        dataset_as_train = anomaly_score(dataset_norm_train, dataset_anom_train, known_outliers, contamination_rate)
    
    input_len = np.product(dataset_ae_train[0][0].shape)


    for run in range(runs):

        # Construct data loaders (re-init for each run)

        loader_ae = DataLoader(dataset_ae_train, batch_size, True, num_workers=num_workers)
        loader_anomaly_score = DataLoader(dataset_as_train, batch_size, True, num_workers=num_workers)

        loader_eval = DataLoader(dataset_eval, batch_size, False, num_workers=num_workers)

        # Init and train autoencoder

        autoencoder = ae_model_class(input_len)

        trainer = pl.Trainer(default_root_dir=checkpoint_ae_dir, max_epochs = AEepochs, callbacks=[auc()], gpus=gpus)

        trainer.fit(autoencoder, loader_ae, val_dataloaders=loader_eval)

        # Init and train anomaly score model, using pretrained autoencoder

        anomaly_score_model = anomaly_score_model_class(ckpt_path=get_ckpt_path(checkpoint_ae_dir), ae_hidden=ae_hidden, input_len=input_len, AEmodel_class=ae_model_class)

        trainer = pl.Trainer(default_root_dir=checkpoint_as_dir, max_epochs = ASepochs, callbacks=[auc()], gpus=gpus)
        trainer.fit(anomaly_score_model,train_dataloader=loader_anomaly_score, val_dataloaders=loader_eval)

        # Collect auc scores from anomaly score model and write to Details.txt

        roc,pr = anomaly_score_model.get_aucs(return_current_vals=True)
        auc_rocs.append(roc)
        auc_prs.append(pr)

        write_details(res_dir, auc_rocs=auc_rocs[-1], auc_prs=auc_prs[-1])
    write_details(res_dir, auc_rocs_mean_std=[np.array(auc_rocs).mean(),np.array(auc_rocs).std()], auc_prs_mean_std=[np.array(auc_prs).mean(),np.array(auc_prs).std()], time_to_run=(time()-time0))




