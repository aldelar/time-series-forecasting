import os
import pandas as pd
import numpy as np

def cleanse_file(input_file_path, output_file_path):
    print("Cleansing file: " + input_file_path)
    input_file = open(os.path.join(input_file_path), 'r')
    df = pd.read_csv(input_file)
    df.replace([np.inf, - np.inf], np.nan, inplace = True)
    output_file = open(os.path.join(output_file_path), 'w')
    df.to_csv(output_file, index=False,line_terminator='\n')
    input_file.close()
    output_file.close()

if __name__ == "__main__":
    cleanse_file('datalake/results/region_FWST_arima_results.csv', 'datalake/results/region_FWST_arima_results_cleansed.csv')
    cleanse_file('datalake/results/state_CA_arima_results.csv', 'datalake/results/state_CA_arima_results_cleansed.csv')