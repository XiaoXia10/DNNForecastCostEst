#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from os.path import join
import os
import argparse


def main(args):
    
    df = pd.read_csv(join(args.data_dir, "df.csv"), index_col=0, parse_dates=True)
    
    vars()["resampled_df_"+args.freq] =  df.resample(args.freq).first()

    
    vars()["resampled_df_"+args.freq].to_csv(f"{args.data_dir}/{args.freq}/df.csv")
    
    return vars()["resampled_df_"+args.freq]


if __name__ == "__main__":
    
    dataset = "karst_data" ###[karst_data, sw_data, gw_data]
    freq = "24h" #[1h, 4h, 8h, 12h, 24h]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="path/"+dataset, help="Data dir")
    # parser.add_argument("--save_dir", type=str, default=freq, help="Save Data dir")
    
    parser.add_argument("--freq", type=str, default=freq, help="freq per time step")
    
    args = parser.parse_args()
    if not os.path.exists(f"{args.data_dir}/{args.freq}"):
        os.mkdir(f"{args.data_dir}/{args.freq}")
    
    main(args)
        

