# -*- coding: utf-8 -*-

from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
import pandas as pd
from os.path import join
import numpy as np


def _load_loader_data(save_dir, loader):
   
    yhat = pd.read_csv(join(save_dir, loader+"_predy.csv"))
    realy = pd.read_csv(join(save_dir, loader+"_realy.csv"))
    
    return yhat, realy

def _destandardize_pred(df_path, save_dir, loader):
    
    df = pd.read_csv(df_path, parse_dates=True, index_col=0)
    # df = df.drop(columns=["milamont"])
    
    std = df.std()
    mean = df.mean()
    
    yhat, realy = _load_loader_data(save_dir,loader)
    
    yhat = (yhat*std.values)+mean.values
    realy = (realy*std.values)+mean.values

    time = np.load(join(save_dir, loader+"_time_y.npy"), allow_pickle=True)
    time = pd.to_datetime(time.flatten())
           
    yhat.index = time
    realy.index = time
       
    return yhat, realy

def get_metrics():
    
    loader = "test"
    forecasts = [3, 6, 12]
    timesteps = ["D","12H","8H","4H","H"]
    
    # path = r"G:\My Drive\Neuchatel_Project\Betteraz\python_codes\LSTM_auto"
    # name = join(path, "LSTM_model_metrics_masl.csv")
    
    path = r"G:\My Drive\Neuchatel_Project\Betteraz\python_codes\improved_graph_wavenet"
    name = join(path, "GWN_model_metrics_masl.csv")
    
    f = open(name, 'w')
    f.write("loader, Timestep, Forecast, RMSE, R2, MAPE, MAE \n" )

    for timestep in timesteps:
    
        df_path = "data/betteraz_"+str(timestep)+"_sims.csv"
        
        for forecast in forecasts:
            
            save_path = join(path, "experiment_"+str(timestep)+"_"+str(forecast))
            
            yhat, realy = _destandardize_pred(df_path, save_path, loader)
    
            vars()[loader+'_RMSE'] = root_mean_squared_error(realy, yhat)
            vars()[loader+'_R2'] = r2_score(realy, yhat)
            vars()[loader+'_MAPE'] = mean_absolute_percentage_error(realy, yhat)
            vars()[loader+'_MAE'] = mean_absolute_error(realy, yhat)
        
            f.write(str(loader)+",")
            f.write(str(timestep)+",")
            f.write(str(forecast)+",")
            f.write(str(vars()[loader+'_RMSE'])+",")
            f.write(str(vars()[loader+'_R2'])+",")
            f.write(str(vars()[loader+'_MAPE'])+",")
            f.write(str(vars()[loader+'_MAE'])+",")
            f.write("\n") 
        
            print(loader, "\n")
            print("RMSE:")
            print(vars()[loader+'_RMSE'])
            print("R2:")
            print(vars()[loader+'_R2'])
            print("MAPE:")
            print(vars()[loader+'_MAPE'])
            print("MAE:")
            print(vars()[loader+'_MAE'], "\n")

    f.close()

get_metrics()
    
