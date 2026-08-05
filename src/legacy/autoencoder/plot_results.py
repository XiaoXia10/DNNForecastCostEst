# -*- coding: utf-8 -*-

import argparse
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from utils import get_shared_arg_parser, destandardize_pred
from os.path import join, dirname, abspath

def main(args):

    yhat = pd.read_csv(join(args.output_dir, "test_predy.csv"), index_col=0, parse_dates=True)
    realy = pd.read_csv(join(args.output_dir, "test_realy.csv"), index_col=0, parse_dates=True)
    uncert =  pd.read_csv(join(args.output_dir, "test_uncert.csv"), index_col=0, parse_dates=True)

    well = 2
    
    uncert_pos = yhat+uncert
    uncert_neg = yhat-uncert
    pos = (uncert_pos.iloc[:,well].to_numpy()).flatten()
    neg = (uncert_neg.iloc[:,well].to_numpy()).flatten()
    
    
    fig = plt.figure(figsize=(20, 10))
    plt.plot(realy.index, realy.iloc[:,well], label="Measured",  linewidth=3)
    plt.plot(yhat.index, yhat.iloc[:,well], label= "Predicted", linewidth=3)
    plt.fill_between(uncert.index, pos, neg, color ="green", alpha = 0.4, label = "Uncertainty")

    plt.title(f"{args.seq}-Time Step Forecast Horizon", fontsize = 35)
    # ax1.set_ylabel("GWL (masl)", fontsize = 25)
    plt.xticks(fontsize=25, rotation=45)
    plt.yticks(fontsize=25)
    plt.legend(fontsize = 25)
    plt.xlabel("Date", fontsize=25)
    plt.ylabel("Groundwater Level (masl)", fontsize=25)
    # fig = plt.figure(figsize=(20, 10))
    # plt.plot(realy['2015-10-01':'2015-12-15'], label="Measured",  linewidth=3)
    # plt.plot(yhat['2015-10-01':'2015-12-15'], label= "Predicted", linewidth=3)
    # plt.fill_between(realy['2015-10-01':'2015-12-15'].index, ((uncert_pos['2015-10-01':'2015-12-15']).to_numpy()).flatten(), ((uncert_neg['2015-10-01':'2015-12-15']).to_numpy()).flatten(), color ="green", alpha = 0.5, label = "Uncertainty")

    # plt.title(f"{args.seq}-Time Steps Forecast Horizon", fontsize = 35)
    # # ax1.set_ylabel("GWL (masl)", fontsize = 25)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.legend(fontsize = 25)


if __name__ == "__main__":
    
    freq = "1h" ##[1h, 4h, 8h, 12h, 24h]


    seq_length_x = 3 #[3,6,9,12]
    seq_length_y = 3
    shift = 3
    
    current_dir = dirname(abspath(__file__))
    base_dir = dirname(current_dir)
    
    data_dir = join(base_dir, "data", freq)
    output_dir = join(data_dir, "timestep"+str(seq_length_x)+str(seq_length_y)+str(shift))
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", type=str, default=data_dir, help="Data directory")
    parser.add_argument("--output_dir", type=str, default=output_dir, help="Data for model training and model output.",)
    
    
    parser.add_argument("--seq", type=str, default=seq_length_y, help="The resampled data frequency")
    # parser.add_argument("--dir", type=str, default="/Users/xl3138/workspaces/data_resolution_study/lstm/"+dataset+"/"+freq, help="Data dir")
    # parser.add_argument("--data_dir", type=str, default="timestep"+str(seq_length_x)+str(seq_length_y)+str(shift), help="Data directory for model training.",)
    
    args = parser.parse_args()
    
    
    main(args)
    
    
    