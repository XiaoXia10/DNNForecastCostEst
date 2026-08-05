# -*- coding: utf-8 -*-

import argparse
import os 
from durbango import pickle_save

def get_shared_arg_parser():
    timestep = "D" # The time between each step, H=hour, 4H=4hours
    seq_length_x = 12
    seq_length_y = 12
    shift = 12
    
    print(timestep)
    print(seq_length_x)
  
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lstm', type=str, default=False, help='If True use the LSTM autoencoder, if False use GRU autoencoder')
    # parser.add_argument('--gru', type=str, default=False, help='If true use the GRU autoencoder')
    
    # Paths of input and output data
    parser.add_argument("--df", type=str, default="data/betteraz_"+str(timestep)+"_sims.csv", help="Input data.")
    parser.add_argument("--save_dir", type=str, default="experiment_GRU_"+str(timestep)+"_"+str(seq_length_x)+str(seq_length_y)+str(shift), help="Output prediction data directory.")
   
    parser.add_argument("--save_model_name", type=str, default= "trained_lstm_model.h5", help="Save trained model name")
    parser.add_argument("--plot_save", type=str, default= r"G:/My Drive/Plots_compare_manuscript/"+str(timestep)+"_GRU_auto", help="Save plot path")
  
    # These are hyperparameters that you must tune
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs') # Keep this value high, it doesn't matter.
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size') 
    parser.add_argument('--latent_dim', type=int, default=300, help='Latent dimension') 
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout') 
    parser.add_argument('--recurrent_dropout', type=float, default=0.7, help='Recurrent dropout') # high dropouts create a cyclic prediction 
        
       
    parser.add_argument('--n_iters', default=None, help='quit after this many iterations')
    parser.add_argument('--es_patience', type=int, default=20, help='quit if no improvement after this many iterations')
    
    # Data preparation
    parser.add_argument("--seq_length_x", type=int, default=seq_length_x, help="X Sequence Length.",)
    parser.add_argument("--seq_length_y", type=int, default=seq_length_y, help="Y Sequence Length.",)
    parser.add_argument("--shift", type=int, default=shift, help="Default is seq_length_x", ) # this is a sequence window shift
    parser.add_argument("--timestep", type=str, default=timestep, help="Timestep", )


    return parser
