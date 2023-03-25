# Source Label Adaptation
The official Pytorch implementation of "Semi-Supervised Domain Adaptation with Source Label Adaptation" accepted by CVPR 2023. Check more details of this work in our paper: [[Arxiv]](https://arxiv.org/abs/2302.02335).

## Python version & Packages

`python==3.8.13`

```
configargparse==1.5.3
torch==1.12.0
torchvision==0.13.0
tensorboard==2.9.0
Pillow==9.0.1
numpy==1.22.3
```

## Data Preparation

### Supported Datasets

Currently, we support the following three datasets:

- [DomainNet](http://ai.bu.edu/M3SDA/)
- [OfficeHome](https://www.hemanthdv.org/officeHomeDataset.html)
- [Office31](https://faculty.cc.gatech.edu/~judy/domainadapt/)

### Dataset Architecture

The dataset is organized into directories, as shown below:

```
- dataset_dir
    - dataset_name
        - domain 1
        - ...
        - domain N
        - text
            - domain 1
                - all.txt
                - train_1.txt
                - train_3.txt
                - test_1.txt
                - test_3.txt
                - val.txt
            - ...
            - domain N
    - ...
```

### Download and Preparation

Before running the data preparation script, make sure to update the configuration file in `data_preparation/dataset.yaml` with the correct settings for your dataset. In particular, you will need to update the `dataset_dir` variable to point to the directory where your dataset is stored.

```
dataset_dir: /path/to/dataset
```

To download and prepare one of these datasets, run the following command:

```
python data_preparation/data_preparation.py --dataset <DATASET>
```

Replace <DATASET> with the name of the dataset you want to prepare (e.g. DomainNet, OfficeHome, or Office31). This script will download the dataset (if necessary) and extract the text data which specify the way to split training, validation, and test sets. The resulting data will be saved in the format described above.

After running the data preparation script, you should be able to use the resulting data files in this repository.


## Running the model
    
To run the main Python file, use the following command:

```
python --method MME_LC --source 0 --target 1 --seed 1102 --num_iters 10000 --shot 3shot --alpha 0.3 --update_interval 500 --warmup 500 --T 0.6
```
    
This command runs the MME + SLA model on the 3-shot A -> C Office-Home dataset, with the specified hyperparameters. You can modify the command to run different experiments with different hyperparameters or on different datasets.

## Acknowledgement

This readme file is partially generated by [ChatGPT](https://chat.openai.com/chat).

This code is partially based on [MME](https://github.com/VisionLearningGroup/SSDA_MME), [CDAC](https://github.com/lijichang/CVPR2021-SSDA) and [DeepDA](https://github.com/jindongwang/transferlearning/tree/master/code/DeepDA).

The backup urls for OfficeHome, Office31 are provided at [here](https://github.com/jindongwang/transferlearning/blob/master/data/dataset.md).
