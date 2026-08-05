#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from os.path import join
import os
import argparse
from os.path import join, dirname, abspath

def get_parsers():
    
    freq = "12h" #[1h, 4h, 8h, 12h, 24h]
    
    current_dir = dirname(abspath(__file__))
    base_dir = dirname(current_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default= join(base_dir, "data"), help="Data dir")
    parser.add_argument("--save_dir", type=str, default=freq, help="Save Data dir")
    
    parser.add_argument("--freq", type=str, default=freq, help="freq per time step")
    
 
    args = parser.parse_args()
    
    return args
    
def main():
    
    args = get_parsers()
  
    if not os.path.exists(f"{args.data_dir}/{args.freq}"):
        os.mkdir(f"{args.data_dir}/{args.freq}")
      
  
    df = pd.read_csv(join(args.data_dir, "betteraz_h_data.csv"), index_col=0, parse_dates=True)
    
    vars()["resampled_df_"+args.freq] =  df.resample(args.freq).first()

    
    vars()["resampled_df_"+args.freq].to_csv(f"{args.data_dir}/{args.freq}/df.csv")
    
    return vars()["resampled_df_"+args.freq]



if __name__ == "__main__":
    
    
    df = main()
        