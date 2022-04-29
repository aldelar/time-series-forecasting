# time-series-forecasting

This use case trains a set of time series using AutoArima, generating one model per time series, forecasting, and evaluating performance of each time series.

This repo is leveraging:
- Azure Machine Learning (AML)
- AML CLI v2 for the job definitions (yml)

### Setup 

NOTE: you can change these defaults names by updating the CLI v2 yml files
- Input training data to be located into an AML DataStore named 'datalake', in a container name 'ts_train'
- Output training evaluation results will be stored in the 'datalake' DataStore under the 'ts_eval' container
- A compute cluster named 'cpu-cluster' defined in your AML workspace

Input file structure: ts_level_1,ts_level_2,datetime,[features],target
features are optional

Ex of training data set:

    State,SKU,Year-Month,Units
    CA,D100,2020-01,5
    ...
    CA,D100,2021-12,2
    CA,D200,2020-01,3
    ...
    CA,D200,2021-12,11

Example job parameters:

    train_folder:
        type: uri_folder
        path: azureml://datastores/datalake/paths/ts_train
    train_file: "train_region.csv"
    ts_level_1: "Region"
    ts_level_2: "Part Number"
    ts_target: "Units sold per Part"
    ts_forecast_horizon: 12

The concept of AML CLI v2 is that you can build and run your code locally and then execute that same script in Azure w/o any code change.

Run training locally:

    conda env create -f src/forecast/auto_arima_conda.yml
    conda activate time-series-forecasting

    python src/forecast_auto_arima.py --train_folder=datalake/train --train_file=train_region.csv --eval_folder=datalake/eval --ts_level_1=Region --ts_level_2="Part Number" --ts_target="Units sold per Part" --ts_test_horizon=12 --ts_forecast_horizon=12

Run training in AML:

Make sure you have installed the Azure CLI and then that you have the AML extension installed

    az extension add -n ml

Once that's setup, this is how you'd run the same job as above in AML:

    az ml job create -f src/forecast/auto_arima_job.yml

You can also compose jobs into pipeline, for instance here, we have composed two independent jobs in the same pipeline training from 2 different training files (say one is about a set of time series with the level1 being 'Region', the other 'State'):

    az ml job create -f src/forecast/auto_arima_pipeline.yml

Note that in both cases, an experiment named 'time-series-forecasting' will host these jobs run in AML.