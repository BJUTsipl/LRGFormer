#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from  dehamer_model1027g4 import dehamer
from argparse import ArgumentParser

# from  train_data_aug import TrainData #############
from val_data_train import ValData_train
from datasetsio  import TrainDatasetFromFolder2,TrainDatasetFromFolder3,TrainDatasetFromFolder2,TestDatasetFromFolder1
# from val_data import ValData

def parse_args():
    """Command-line argument parser for training."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of dehamer from Guo et al. (2022)')

    # Data parameters
    parser.add_argument('-d', '--dataset-name', help='name of dataset',choices=['NH', 'dense', 'indoor','outdoor'], default='outdoor')
    parser.add_argument('-t', '--train-dir', help='training set path', default='./../data/train')
    parser.add_argument('-v', '--valid-dir', help='test set path', default='./../data/valid')
    parser.add_argument('--ckpt-save-path', help='checkpoint save path', default='./ckpts')
    parser.add_argument('--ckpt-overwrite', help='overwrite model checkpoint on save', action='store_true')
    parser.add_argument('--ckpt-load-path', help='start training with a pretrained model',default='.../ckpts/outdoor/dehamer-epoch11-0.01483.pt')
    parser.add_argument('--report-interval', help='batch report interval', default=1, type=int)
    parser.add_argument('-ts', '--train-size',nargs='+', help='size of train dataset',default=[800,1200], type=int) 
    parser.add_argument('-vs', '--valid-size',nargs='+', help='size of valid dataset',default=[800,1200], type=int)  

    # Training hyperparameters 
    parser.add_argument('-lr', '--learning-rate', help='learning rate', default=0.0002, type=float)
    parser.add_argument('-a', '--adam', help='adam parameters', nargs='+', default=[0.9, 0.99, 1e-8], type=list)
    parser.add_argument('-b', '--batch-size', help='minibatch size', default=16, type=int)
    parser.add_argument('-e', '--nb-epochs', help='number of epochs', default=100, type=int)
    parser.add_argument('-l', '--loss', help='loss function', choices=['l1', 'l2'], default='l1', type=str)
    parser.add_argument('--cuda', help='use cuda', action='store_true')
    parser.add_argument('--plot-stats', help='plot stats after every epoch', action='store_true') 
    parser.add_argument('-s', '--seed', help='fix random seed', type=int)
    parser.add_argument('-c', '--crop-size', help='random crop size', default=128, type=int)#224
    # parser.add_argument('--addition_loss', type=str, default='VGG')
    # parser.add_argument('--lam_VGG', type=float, default=0.1)
    # parser.add_argument('--depth_loss', type=bool, default=True)
    # parser.add_argument('--lam_DepthLoss', type=float, default=0.1)
 
    return parser.parse_args()

 
if __name__ == '__main__':    
    
    
    # Parse training parameters  
    params = parse_args()


    # import pdb;
    # pdb.set_trace()
# --- Load training data and validation/test data --- #

#     train_loader = DataLoader(TrainData(params.dataset_name, params.train_size, params.train_dir),
#                               batch_size=params.batch_size, shuffle=True, num_workers=60)
#     valid_loader = DataLoader(ValData_train(params.dataset_name, params.valid_size, params.valid_dir),
#                               batch_size=params.batch_size, shuffle=False, num_workers=60)
    "HAZY"
#     train_loader = DataLoader(TrainDatasetFromFolder2('.../clear/cleary',
#                                                     '.../haze/hazy35y',  crop_size=128),
#                             batch_size=16, shuffle=True)
#     train_loader = DataLoader(TrainDatasetFromFolder2('.../dehazedata/ITS/clear',
#                                                       '.../dehazedata/ITS/hazy',
#                                                       crop_size=256), batch_size=1, shuffle=True)
#
#     valid_loader = DataLoader(TestDatasetFromFolder1('.../dehazedata/SOTS-I'),
#                               # real test-rrrrrr'.../Raindatasets/rain_data_test_Light-R100L'
#                               batch_size=1, shuffle=False, num_workers=1)
#     train_loader = DataLoader(
#         TrainDatasetFromFolder2('.../OTS/cleary',
#                                 '.../OTS/hazy35y',
#                                 crop_size=128), batch_size=16, shuffle=True)
#
#     valid_loader = DataLoader(TestDatasetFromFolder1('.../reside-outdoor/SOTS-O'),
#                               # real test-rrrrrr'.../rain_data_test_Light-R100L'
#                               batch_size=1, shuffle=False, num_workers=1)
#
    "RAIN"
    # train_loader = DataLoader(TrainDatasetFromFolder2(
    #     '.../dataset/Weather/Snow/Snow100K-training/all/gt',
    #     '.../dataset/Weather/Snow/Snow100K-training/all/synthetic',
    #     crop_size=128), batch_size=24, shuffle=True)
    #
    # valid_loader = DataLoader(TestDatasetFromFolder1(
    #     '.../dataset/Weather/Snow/Snow100K-testset/test/Snow100K-L'),
    #     batch_size=1, shuffle=False, num_workers=1)

    # train_loader = DataLoader(TrainDatasetFromFolder2(  '.../dataset/Weather/Rain/Raindrop/train/target',
    #     '.../dataset/Weather/Rain/Raindrop/train/input',
    #     crop_size=128), batch_size=16, shuffle=True)
    #
    # valid_loader = DataLoader(TestDatasetFromFolder1(
    #     '.../dataset/Weather/Rain/Raindrop/test_b/test_b'),
    #     batch_size=1, shuffle=False, num_workers=1)

    "SNOW"
    train_loader = DataLoader(  TrainDatasetFromFolder2('.../snowdataset/Snow100K-training/all/gt',
                                 '.../snowdataset/Snow100K-training/all/synthetic',
                                crop_size=128),   batch_size=16, shuffle=True)
    valid_loader = DataLoader(TestDatasetFromFolder1('.../snowdataset/testL'),
                               batch_size=1, shuffle=False, num_workers=1)
    # Initialize model and train 
    dehamer = dehamer(params, trainable=True)
    dehamer.train(train_loader, valid_loader)
