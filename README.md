# FEAWAD_Reproducibility_Study
Reproducing the results published in Feature Extracting Autoencoders for Weakly Supervised Anomaly Detection (FEAWAD)

# Motivation 
I am interested in applying techniques from the field of Continual Learning to enable a more versatile and adaptive anomaly detection system.
First I am working on reproducing results published in FEAWAD [1], with a view to extending this to a continual learning setting.

# This Repo

This repo serves two functions: 
    First, to run the original code as is, with only minor adjustments to reduce user input (data_dim function in FEAWAD_Original)
    Second, re-build method in PyTorch since this is my prefered package for ML. Compare results.
When the reproducibility of these results are confirmed I will move on to implementing a Continual Learning environment and methods.

# Original Code

To run original code, install the environment packages described in requirements_original.txt
run as described in the original repo (linked below), e.g.
    python FEAWAD_Unchanged.py --data_set=arrhythmia_normalization,cardio_normalization --runs=10 --epochs=100  

FEAWAD repo:
    https://github.com/yj-zhou/Feature_Encoding_with_AutoEncoders_for_Weakly-supervised_Anomaly_Detection

# My Reproduction

My reproduction code main.py accepts original datasets using the original dataloading procedures, as well as MNIST, Spoken Digits (SPMNIST) and (soon) DCASE 2020 Task 2. 

To run this code, install the environment packages described in requirements_reproduction.txt. 
Or use a python3.7 environment and pip install the following:
    numpy==1.21.5
    scikit-learn==1.0.2
    scipy==1.7.3
    torchvision==0.11.3
    imageio==2.16.1
    pytorch-lightning==1.5.10


Run one dataset at a time, for instance
    python main.py --dataset=MNIST --ASepochs=30 --AEepochs=30 --runs=10

    You'll need to change the root variable to point to your dataset root, for any dataset that isn't in the git repo

[1] Yingjie Zhou, Xucheng Song, Yanru Zhang, Fanxing Liu, Ce Zhu and Lingqiao Liu. Feature Encoding with AutoEncoders for Weakly-supervised Anomaly Detection, IEEE Transactions on Neural Networks and Learning Systems, 2021.

