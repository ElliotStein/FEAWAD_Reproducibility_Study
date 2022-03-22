# FEAWAD_Reproducibility_Study
Reproducing the results published in Feature Extracting Autoencoders for Weakly Supervised Anomaly Detection (FEAWAD)

I am interested in applying techniques from the field of Continual Learning to enable a more versatile and adaptive anomaly detection system.
First I am working on reproducing results published in FEAWAD [1], with a view to extending this to a continual learning setting.

This repo serves two functions: 
    First, to run the original code as is, with only minor adjustments to reduce user input (data_dim function in FEAWAD_Original)
    Second, re-build method in PyTorch since this is my prefered package for ML. Compare results.
When the reproducibility of these results are confirmed I will move on to implementing a Continual Learning environment and methods.

To run original code, install the environment packages described in requirements_original.txt
run as described in the original repo (linked below), e.g.
    python FEAWAD_Unchanged.py --data_set=arrhythmia_normalization,cardio_normalization --runs=10 --epochs=100  

FEAWAD repo:
    https://github.com/yj-zhou/Feature_Encoding_with_AutoEncoders_for_Weakly-supervised_Anomaly_Detection


[1] Yingjie Zhou, Xucheng Song, Yanru Zhang, Fanxing Liu, Ce Zhu and Lingqiao Liu. Feature Encoding with AutoEncoders for Weakly-supervised Anomaly Detection, IEEE Transactions on Neural Networks and Learning Systems, 2021.

