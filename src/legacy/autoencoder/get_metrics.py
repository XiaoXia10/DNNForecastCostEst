#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd 
from os.path import join 
from utils import destandardize_pred, get_shared_arg_parser
import argparse
import numpy as np
import pickle
from permetrics.regression import RegressionMetric

def get_metrics(args):
    
    test_pred = pd.read_csv(join(args.output_dir, "test_realy.csv"), index_col=0, parse_dates=True)
    test_real = pd.read_csv(join(args.output_dir, "test_predy.csv"), index_col=0, parse_dates=True)
    
    pred = destandardize_pred(args, test_pred)
    true = destandardize_pred(args, test_real)
    ####### Evaluate each well seperately #########
    list_metrics = ["RMSE", "MAE", "NSE", "KGE"]
    evaluator = RegressionMetric(true.to_numpy(), pred.to_numpy())
    results = evaluator.get_metrics_by_list_names(list_metrics)
    
    ######## Save results into a file ########
    ######## This is a pickle file, can only be open with the same environment
    with open(join(args.output_dir, "metric.json"), 'wb') as file:
        pickle.dump(results, file)
        
    f = open(join(args.output_dir, "metric.csv"), 'w')
    f.write ("RMSE (l/s), MAE (l/s), NSE, KGE \n" )  
    
    ###### metric mean of all wells
    for metric in list_metrics:
        vars()[metric] = results[metric].mean()
        if metric != "KGE":
            f.write(str(vars()[metric])+",")
        else:
            f.write(str(vars()[metric]))
            f.write("\n") 
            f.close() 
        
    return results

if __name__ == "__main__":
    
    args =get_shared_arg_parser()
    
    results = get_metrics(args)