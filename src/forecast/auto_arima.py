import os, time, math, argparse

import pandas as pd
import numpy as np

import pmdarima as pm
from pmdarima.arima import ADFTest

import warnings
warnings.filterwarnings("ignore")

# compute accuracy metrics
def compute_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None],actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None],actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 
            'corr':corr, 'minmax':minmax})

#
def train_models(train_folder,train_file,eval_folder,ts_level_1,ts_level_2,ts_target,ts_test_horizon,ts_forecast_horizon):
    
    # training data location
    data_file_path = os.path.join(train_folder,train_file)
    # eval data location
    os.makedirs(eval_folder, exist_ok=True)
    eval_file_path = os.path.join(eval_folder, ts_level_1 + '_eval.csv')
    eval_file = open(eval_file_path, 'w')
    eval_file.write(f"{ts_level_1},{ts_level_2},train_count,train_std,train_mean,train_min,train_max,adf_test_pval,adf_test_sig,training_time_in_s,forecast_time_in_s,acc_mape,acc_me,acc_mae,acc_mpe,acc_rmse,acc_corr,acc_minmax\n")
    #
    print(f"==> train_models()\n    in : {data_file_path})\n    out: {eval_file_path}")

    # load training data
    training_df = pd.read_csv(data_file_path)

    # get unique level1 values
    level_1_values = training_df[ts_level_1].unique()
    level_1_values_len = len(level_1_values)
    for level_1 in level_1_values:

        # filter training_df by level1
        level_1_df = training_df[training_df[ts_level_1] == level_1]
        # drop level1 column
        level_1_df = level_1_df.drop([ts_level_1], axis=1)

        # get unique level2 values for level1 time series
        level_2_values = level_1_df[ts_level_2].unique()
        level_2_values_len = len(level_2_values)
        i = 0
        for level_2 in level_2_values:
            level_2_df = level_1_df[level_1_df[ts_level_2] == level_2]
            # drop ts_level_2 column
            level_2_df = level_2_df.drop([ts_level_2], axis=1)
            len_train = len(level_2_df[ts_target])-ts_test_horizon
            train_df = level_2_df[:len_train]
            test_df = level_2_df[len_train:]

            # train stats
            train_stats = train_df[ts_target].describe()
            train_count = train_stats['count']
            train_mean = train_stats['mean']
            train_min = train_stats['min']
            train_max = train_stats['max']
            train_std = train_stats['std']

            # augmented Dickey–Fuller test
            adf_test = ADFTest(alpha = 0.05)
            adf_test_results = adf_test.should_diff(train_df[ts_target])
            adf_test_pval = adf_test_results[0]
            adf_test_sig = adf_test_results[1]
            #print(f"adf_test_pval: {adf_test_pval}")
            #print(f"adf_test_sig: {adf_test_sig}")

            # autoarima
            training_start_time = time.time()
            model = pm.auto_arima(train_df[ts_target], start_p=1, start_q=1,
                test='adf',       # use adftest to find optimal 'd'
                max_p=3, max_q=3, # maximum p and q
                m=1,              # frequency of series
                d=None,           # let model determine 'd'
                seasonal=False,   # Seasonality
                start_P=0, 
                D=0, 
                trace=False,
                error_action='ignore',  
                suppress_warnings=True, 
                stepwise=True)
            training_end_time = time.time()
            training_time_in_s = training_end_time - training_start_time
            #print(f"training_time_in_s: {training_time_in_s}")
            #print(model.summary())
            #print(model.params())

            # forecast
            forecast_start_time = time.time()
            n_periods = 12
            fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
            forecast_end_time = time.time()
            forecast_time_in_s = forecast_end_time - forecast_start_time
            #print(f"forecast_time_in_s: {forecast_time_in_s}")

            # compute model accuracy
            forecast = fc
            actual = np.array(test_df[ts_target])
            accuracy = compute_accuracy(forecast, actual)
            #print(f"forecast: {forecast}")
            #print(f"actual: {actual}")
            #print(f"accuracy: {accuracy}")

            # output results
            i += 1
            print(f"{i}/{level_2_values_len} Trained {level_1}/{level_2} => train_mean: {train_mean} | train_std: {train_std} | mape: {round(accuracy['mape']*100, 0)}% | trained in {round(training_time_in_s,2)} seconds")

            # save results
            eval_file.write(f"{level_1},{level_2},{train_count},{train_std},{train_mean},{train_min},{train_max},{adf_test_pval},{adf_test_sig},{training_time_in_s},{forecast_time_in_s},{accuracy['mape']},{accuracy['me']},{accuracy['mae']},{accuracy['mpe']},{accuracy['rmse']},{accuracy['corr']},{accuracy['minmax']}\n")

    # close output file
    eval_file.close()

# parse script arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_folder", type=str)
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--eval_folder", type=str)
    parser.add_argument("--ts_level_1", type=str)
    parser.add_argument("--ts_level_2", type=str)
    parser.add_argument("--ts_target", type=str)
    parser.add_argument("--ts_test_horizon", type=int)
    parser.add_argument("--ts_forecast_horizon", type=int)
    return parser.parse_args()

# main
if __name__ == "__main__":
    args = parse_args()
    train_models(train_folder=args.train_folder,
                train_file=args.train_file,
                eval_folder=args.eval_folder,
                ts_level_1=args.ts_level_1,
                ts_level_2=args.ts_level_2,
                ts_target=args.ts_target,
                ts_test_horizon=args.ts_test_horizon,
                ts_forecast_horizon=args.ts_forecast_horizon
                )