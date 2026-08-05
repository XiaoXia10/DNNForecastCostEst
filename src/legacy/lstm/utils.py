#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from os.path import join
import argparse
from durbango import pickle_save
import pandas as pd

def destandardize_pred(args, df):
    
    data = pd.read_csv(join(args.dir, "df.csv"), index_col=0, parse_dates=True)
    std = data.std()
    mean = data.mean()
    
    df_copy = df.copy()
    df_destand = (df_copy*std.values)+mean.values
   
    return df_destand

def standardize_df(args):
    
    data = pd.read_csv(join(args.dir, "df.csv"), index_col=0, parse_dates=True)
    std = data.std()
    mean = data.mean()
    
    df_std = (data-mean.values)/std.values
    
    return df_std

def log_df(df):

    df = np.log10(df)
    
    return df

# def exp10_pred(args, df):
    
#     data = pd.read_csv(join(args.dir, "df.csv"), index_col=0, parse_dates=True)
    
#     yhat = np.power(10.0, yhat)
#     realy = np.power(10.0, realy)
    
#     yhat = pd.DataFrame(yhat)

    
#     yhat.index = time
#     realy.index = time
    
#     return yhat, realy
    

def data_loader(args, loader):
    
    x = np.load(join(f"{args.dir}/{args.data_dir}", "x_"+loader+".npy"))
    y = np.load(join(f"{args.dir}/{args.data_dir}", "y_"+loader+".npy"))    

    x = torch.tensor(x).float()
    y = torch.tensor(y).float()

    valid = ~(torch.isnan(x).any(dim=(1, 2)) | torch.isnan(y).any(dim=(1, 2)))
    x, y = x[valid], y[valid]

    train_dataset = TensorDataset(x, y)
    
    if loader == "train":
        
        data_loader = DataLoader(
                                  dataset=train_dataset, 
                                  batch_size=args.batch_size, 
                                  shuffle=True
                                 )
        
    else:
        data_loader = DataLoader(
                                  dataset=train_dataset, 
                                  batch_size=args.batch_size, 
                                  shuffle=False
                                 )
    return data_loader

def get_shared_arg_parser():
    
    freq = "4h" ##[1h, 4h, 8h, 12h, 24h]
    forecast = 12
    
    ###[karst_data, sw_data, gw_confined_data, gw_unconfined_data]
    dataset = "karst_data"
    
    seq_length_x = forecast #[3,6,9,12]
    seq_length_y = forecast
    shift = forecast
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size') 
    parser.add_argument('--input_dim', type=int, default=1, help='This is the number of input features in the data') 
    parser.add_argument('--output_dim', type=int, default=1, help='This is the number of output features from model prediction') 
    parser.add_argument('--hidden_dim', type=int, default=8, help='Number of hidden dimensions') 
    parser.add_argument('--num_layers', type=int, default=5, help='Number of model layers')
    
    parser.add_argument('--dropout_rate', type=float, default=0.4, help='Recurrent dropout') 
    
    parser.add_argument('--patience', type=int, default=15, help='quit if no improvement after this many iterations')
    parser.add_argument('--num_real', type=int, default=150, help='Number of MC dropout forward passes for uncertainty estimation')
    parser.add_argument('--mc_dropout_rate', type=float, default=0.3, help='Dropout rate applied during MC dropout inference')
    
    parser.add_argument("--dir", type=str, default="path/"+dataset+"/"+freq, help="Data dir")
    
    parser.add_argument("--data_dir", type=str, default="timestep"+str(seq_length_x)+str(seq_length_y)+str(shift), help="Data directory for model training.",)
    
    args = parser.parse_args()
    # pickle_save(args, f'{args.dir}/{args.data_dir}/args.pkl')
    
    return args