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
def train_models(train_folder,train_file,tier_level,eval_folder):
    
    # training data location
    data_file_path = os.path.join(train_folder,train_file)
    # eval data location
    print(f"eval_folder: {eval_folder}")
    os.makedirs(eval_folder, exist_ok=True)
    eval_file_path = os.path.join(eval_folder, tier_level + '_eval.csv')
    eval_file = open(eval_file_path, 'w')
    eval_file.write("tier_level,tier,sku,train_count,train_std,train_mean,train_min,train_max,adf_test_pval,adf_test_sig,training_time_in_s,forecast_time_in_s,acc_mape,acc_me,acc_mae,acc_mpe,acc_rmse,acc_corr,acc_minmax\n")
    #
    print(f"==> train_models()\n    in : {data_file_path})\n    out: {eval_file_path}")

    # load training data
    training_df = pd.read_csv(data_file_path)

    # get unique tiers
    tiers = training_df[tier_level].unique()
    tiers_len = len(tiers)

    for tier in tiers:

        #
        tier_df = training_df[training_df[tier_level] == tier]
        # data engineering
        tier_df = tier_df.drop([tier_level], axis=1)

        # get unique items from training_df 'Part Number' column
        skus = tier_df['Part Number'].unique()

        # compute model for each sku
        i = 0
        skus_len = len(skus)
        for sku in skus:
            df = tier_df[tier_df['Part Number'] == sku]
            df = df.drop(['Part Number'], axis=1)
            len_train = len(df['Units sold per Part'])-12
            train = df[:len_train]
            test = df[len_train:]

            # train stats
            train_stats = train['Units sold per Part'].describe()
            train_count = train_stats['count']
            train_mean = train_stats['mean']
            train_min = train_stats['min']
            train_max = train_stats['max']
            train_std = train_stats['std']

            # augmented Dickey–Fuller test
            adf_test = ADFTest(alpha = 0.05)
            adf_test_results = adf_test.should_diff(train['Units sold per Part'])
            adf_test_pval = adf_test_results[0]
            adf_test_sig = adf_test_results[1]
            #print(f"adf_test_pval: {adf_test_pval}")
            #print(f"adf_test_sig: {adf_test_sig}")

            # autoarima
            training_start_time = time.time()
            model = pm.auto_arima(train['Units sold per Part'], start_p=1, start_q=1,
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
            actual = np.array(test['Units sold per Part'])
            accuracy = compute_accuracy(forecast, actual)
            #print(f"forecast: {forecast}")
            #print(f"actual: {actual}")
            #print(f"accuracy: {accuracy}")

            # output results
            i += 1
            print(f"{i}/{skus_len} Trained {tier_level}/{tier}/{sku} => train_mean: {train_mean} | train_std: {train_std} | mape: {round(accuracy['mape']*100, 0)}% | trained in {round(training_time_in_s,2)} seconds")

            # save results
            eval_file.write(f"{tier_level},{tier},{sku},{train_count},{train_std},{train_mean},{train_min},{train_max},{adf_test_pval},{adf_test_sig},{training_time_in_s},{forecast_time_in_s},{accuracy['mape']},{accuracy['me']},{accuracy['mae']},{accuracy['mpe']},{accuracy['rmse']},{accuracy['corr']},{accuracy['minmax']}\n")

    # close output file
    eval_file.close()

# parse script arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_folder", type=str)
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--tier_level", type=str)
    parser.add_argument("--eval_folder", type=str)
    return parser.parse_args()

# main
if __name__ == "__main__":
    args = parse_args()
    train_models(train_folder=args.train_folder,
                train_file=args.train_file,
                tier_level=args.tier_level,
                eval_folder=args.eval_folder)