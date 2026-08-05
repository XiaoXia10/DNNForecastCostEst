# -*- coding: utf-8 -*-

import argparse
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from utils import get_shared_arg_parser, destandardize_pred


def main(args):

    yhat = pd.read_csv(join(f"{args.dir}/{args.data_dir}", "test_predy.csv"), index_col=0, parse_dates=True)
    realy = pd.read_csv(join(f"{args.dir}/{args.data_dir}", "test_realy.csv"), index_col=0, parse_dates=True)
    uncert =  pd.read_csv(join(f"{args.dir}/{args.data_dir}", "test_uncert.csv"), index_col=0, parse_dates=True)
    
    uncert_pos = yhat+uncert
    uncert_neg = yhat-uncert
    
    fig = plt.figure(figsize=(20, 10))
    plt.plot(realy.index, realy, label="Measured",  linewidth=3)
    plt.plot(yhat.index, yhat, label= "Predicted", linewidth=3)
    plt.fill_between(uncert.index, (uncert_pos.to_numpy()).flatten(), (uncert_neg.to_numpy()).flatten(), color ="green", alpha = 0.5, label = "Uncertainty")

    plt.title(f"{args.seq}-Time Steps Forecast Horizon", fontsize = 35)
    # ax1.set_ylabel("GWL (masl)", fontsize = 25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    # plt.gca().invert_yaxis()
    plt.legend(fontsize = 25)
    plt.xlabel("Date", fontsize=25)
    plt.ylabel("Groundwater Level (masl)", fontsize=25)
    plt.savefig(join(f"{args.dir}/{args.data_dir}", "forecast_plot.png"), dpi=150, bbox_inches='tight')
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
    
    freq = "4h" ##[1h, 4h, 8h, 12h, 24h]
    forecast = 12
    
    ###[karst_data, sw_data, gw_confined_data, gw_unconfined_data]
    dataset = "karst_data"

    seq_length_x = forecast #[3,6,9,12]
    seq_length_y = forecast
    shift = forecast
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seq", type=str, default=seq_length_y, help="The resampled data frequency")
    parser.add_argument("--dir", type=str, default="path/"+dataset+"/"+freq, help="Data dir")
    parser.add_argument("--data_dir", type=str, default="timestep"+str(seq_length_x)+str(seq_length_y)+str(shift), help="Data directory for model training.",)
    
    args = parser.parse_args()
    
    main(args)
    
    
    