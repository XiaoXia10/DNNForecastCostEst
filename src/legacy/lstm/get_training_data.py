#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from os.path import join
import os
import argparse
from  utils import standardize_df


def _training_data_split(args, df):
    
    
    train_data = df[df.index[0]:args.train_end_time]
    test_data = df[args.test_start_time:df.index[-1]]
    
    train_data.to_csv(join(args.dir, "train_val_data.csv"))
    test_data.to_csv(join(args.dir, "test_data.csv"))
    
    return train_data, test_data

def sequence_data_preparation(args, df):
    
    seq_length_x, seq_length_y, shift = args.seq_length_x, args.seq_length_y, args.shift
    
    num_samples, num_nodes = df.shape
    data = df.to_numpy()
    
    df_time = df.index
    max_t = num_samples - (seq_length_y+shift)
    x, y, time_list = [], [], []
    
    for t in range(0, max_t, shift):  # t is the index of the last observation.
        
        total_window_len = data[t:seq_length_x+seq_length_y+t]
        x.append(total_window_len[:seq_length_x,:])
        y.append(total_window_len[seq_length_x:,:]) 
        
        time_window_len = df_time[t:seq_length_x+seq_length_y+t]
        time_list.append(time_window_len[seq_length_x:]) # Only collecting the time for y dataset
        
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    time = np.stack(time_list, axis=0)
    
    return x, y, time


def main(args):

    df_scaled = standardize_df(args)
    train_data, test_data = _training_data_split(args, df_scaled)
    
    x, y, time = sequence_data_preparation(args, train_data)
    x_test, y_test, time_test= sequence_data_preparation(args, test_data)
    
    num_samples = x.shape[0]
    num_train = round(num_samples * args.train_percent)

    x_train, y_train, time_train = x[:num_train], y[:num_train], time[:num_train]

    x_val, y_val, time_val = x[num_train:], y[num_train:], time[num_train:]
    
    np.save(join( f"{args.dir}/{args.output_dir}", "x_train.npy"), x_train)
    np.save(join( f"{args.dir}/{args.output_dir}", "y_train.npy"), y_train)
    np.save(join( f"{args.dir}/{args.output_dir}", "x_val.npy"), x_val)
    np.save(join( f"{args.dir}/{args.output_dir}", "y_val.npy"), y_val)
    np.save(join( f"{args.dir}/{args.output_dir}", "x_test.npy"), x_test)
    np.save(join( f"{args.dir}/{args.output_dir}", "y_test.npy"), y_test)
    np.save(join( f"{args.dir}/{args.output_dir}", "test_time.npy"), time_test)
    np.save(join( f"{args.dir}/{args.output_dir}", "val_time.npy"), time_val)
    np.save(join( f"{args.dir}/{args.output_dir}", "train_time.npy"), time_train)

    return 0

if __name__ == "__main__":
    
    freq = "4h" ##[1h, 4h, 8h, 12h, 24h]
    forecast = 6  #[3,6,9,12]
    
    ###[karst_data, sw_data, gw_data]
    dataset = "karst_data"
    train_ends = "2013-12-31"
    test_begins = "2014-01-01"
    
    seq_length_x = forecast #[3,6,9,12]
    seq_length_y = forecast
    shift = forecast
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_length_x", type=int, default=seq_length_x, help="X Sequence Length.",)
    parser.add_argument("--seq_length_y", type=int, default=seq_length_y, help="Y Sequence Length.",)
    parser.add_argument("--shift", type=int, default=shift, help="Default is seq_length_x", ) # this is a sequence window shift
    
    parser.add_argument("--output_dir", type=str, default="timestep"+str(seq_length_x)+str(seq_length_y)+str(shift), help="Data for model training.",)
    
    parser.add_argument('--train_percent', type=float, default=0.8, help='The percentage of data used for model training and the rest for validation')

    
    parser.add_argument("--dir", type=str, default=" path/"+dataset+"/"+freq, help="Data dir")
    
    parser.add_argument("--test_start_time", type=str, default=test_begins, help="Start time to end of time series")
    parser.add_argument("--train_end_time", type=str, default=train_ends, help="End time of training data")
    
    args = parser.parse_args()
    
    if not os.path.exists(f"{args.dir}/{args.output_dir}"):
        os.mkdir(f"{args.dir}/{args.output_dir}")
        
    main(args)