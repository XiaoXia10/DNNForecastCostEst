#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from os.path import join
import argparse
from durbango import pickle_save
import pandas as pd
from os.path import join, dirname, abspath

def destandardize_pred(args, df):

    data = pd.read_csv(join(args.data_dir, "train_val_data.csv"), index_col=0, parse_dates=True)
    std = data.std()
    mean = data.mean()

    df_copy = df.copy()
    df_destand = (df_copy*std.values)+mean.values

    return df_destand

def standardize_df(df, mean=None, std=None):

    if mean is None:
        mean = df.mean()
    if std is None:
        std = df.std()

    df_std = (df-mean.values)/std.values

    return df_std, mean, std

def log_df(df):

    df = np.log10(df)
    
    return df



def get_shared_arg_parser():
    
    freq = "24h" # The time between each step, H=hour, 4H=4hours
    seq_length_x = 12
    seq_length_y = 12
    shift = 12
    
    current_dir = dirname(abspath(__file__))
    base_dir = dirname(current_dir)

    data_dir = join(base_dir, "data", freq)
    output_dir = join(data_dir, "timestep"+str(seq_length_x)+str(seq_length_y)+str(shift))
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lstm', type=str, default=False, help='If True use the LSTM autoencoder, if False use GRU autoencoder')
    # parser.add_argument('--gru', type=str, default=False, help='If true use the GRU autoencoder')
    
    # Paths of input and output data
    # parser.add_argument("--df", type=str, default="data/betteraz_"+str(timestep)+"_sims.csv", help="Input data.")
    # parser.add_argument("--save_dir", type=str, default="experiment_GRU_"+str(timestep)+"_"+str(seq_length_x)+str(seq_length_y)+str(shift), help="Output prediction data directory.")
    
    parser.add_argument("--data_dir", type=str, default=data_dir, help="Data directory")
    parser.add_argument("--output_dir", type=str, default=output_dir, help="Data for model training and model output.",)
   
    parser.add_argument("--save_model_name", type=str, default= "best_trained_model.h5", help="Save trained model name")
    # parser.add_argument("--plot_save", type=str, default= r"G:/My Drive/Plots_compare_manuscript/"+str(freq)+"_GRU_auto", help="Save plot path")
  
    # These are hyperparameters that you must tune
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs') # Keep this value high, it doesn't matter.
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size') 
    parser.add_argument('--latent_dim', type=int, default=300, help='Latent dimension') 
    parser.add_argument('--dropout', type=float, default=0.05, help='Dropout') 
    parser.add_argument('--recurrent_dropout', type=float, default=0.3, help='Recurrent dropout') # high dropouts create a cyclic prediction 
        
       
    parser.add_argument('--n_iters', default=None, help='quit after this many iterations')
    parser.add_argument('--es_patience', type=int, default=20, help='quit if no improvement after this many iterations')
    
    # Data preparation
    # parser.add_argument("--seq_length_x", type=int, default=seq_length_x, help="X Sequence Length.",)
    # parser.add_argument("--seq_length_y", type=int, default=seq_length_y, help="Y Sequence Length.",)
    # parser.add_argument("--shift", type=int, default=shift, help="Default is seq_length_x", ) # this is a sequence window shift
    # parser.add_argument("--timestep", type=str, default=timestep, help="Timestep", )

    args = parser.parse_args()
    
    return args
