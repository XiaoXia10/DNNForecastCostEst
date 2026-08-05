#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from os.path import join
from utils import get_shared_arg_parser
import numpy as np
from permetrics.regression import RegressionMetric

def get_metrics(args):

    # Both CSVs are already de-standardized by get_uncertainty.py
    test_real = pd.read_csv(join(f"{args.dir}/{args.data_dir}", "test_realy.csv"), index_col=0, parse_dates=True)
    test_pred = pd.read_csv(join(f"{args.dir}/{args.data_dir}", "test_predy.csv"), index_col=0, parse_dates=True)

    list_metrics = ["RMSE", "MAE", "MSE", "MAPE", "NSE", "KGE"]
    evaluator = RegressionMetric(test_real.to_numpy(), test_pred.to_numpy())
    results = evaluator.get_metrics_by_list_names(list_metrics)

    with open(join(f'{args.dir}/{args.data_dir}', "metric.csv"), 'w') as f:
        f.write("RMSE (l/s), MAE (l/s), MSE (l/s), MAPE, NSE, KGE\n")
        for metric in list_metrics:
            suffix = "\n" if metric == "KGE" else ","
            f.write(str(results[metric]) + suffix)

    return results

if __name__ == "__main__":
    
    args =get_shared_arg_parser()
    
    results = get_metrics(args)