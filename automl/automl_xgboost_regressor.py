# This file has been autogenerated by version 1.40.0 of the Azure Automated Machine Learning SDK.


import numpy
import numpy as np
import pandas as pd
import pickle
import argparse


def setup_instrumentation():
    import logging
    import sys

    from azureml.core import Run
    from azureml.telemetry import INSTRUMENTATION_KEY, get_telemetry_log_handler
    from azureml.telemetry._telemetry_formatter import ExceptionFormatter

    logger = logging.getLogger("azureml.training.tabular")

    try:
        logger.setLevel(logging.INFO)

        # Add logging to STDOUT
        stdout_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stdout_handler)

        # Add telemetry logging with formatter to strip identifying info
        telemetry_handler = get_telemetry_log_handler(
            instrumentation_key=INSTRUMENTATION_KEY, component_name="azureml.training.tabular"
        )
        telemetry_handler.setFormatter(ExceptionFormatter())
        logger.addHandler(telemetry_handler)

        # Attach run IDs to logging info for correlation if running inside AzureML
        try:
            run = Run.get_context()
            parent_run = run.parent
            return logging.LoggerAdapter(logger, extra={"codegen_run_id": run.id, "parent_run_id": parent_run.id})
        except Exception:
            pass
    except Exception:
        pass

    return logger


logger = setup_instrumentation()


def split_dataset(X, y, weights, split_ratio, should_stratify):
    from sklearn.model_selection import train_test_split

    random_state = 42
    if should_stratify:
        stratify = y
    else:
        stratify = None

    if weights is not None:
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
            X, y, weights, stratify=stratify, test_size=split_ratio, random_state=random_state
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=stratify, test_size=split_ratio, random_state=random_state
        )
        weights_train, weights_test = None, None

    return (X_train, y_train, weights_train), (X_test, y_test, weights_test)


def get_training_dataset(dataset_id):
    from azureml.core import Workspace, Dataset

    subscription_id = 'c5ec24ce-9c5f-4da2-bf12-9ca8e9758d60'
    resource_group = 'synapse'
    workspace_name = 'aldelarml'

    workspace = Workspace(subscription_id, resource_group, workspace_name)

    #dataset = Dataset.get_by_name(workspace, name='epi_region_FWST_DS1438_train')
    dataset = Dataset.get_by_id(workspace=workspace, id=dataset_id)
    return dataset.to_pandas_dataframe()


def prepare_data(dataframe):
    from azureml.training.tabular.preprocessing import data_cleaning
    
    logger.info("Running prepare_data")
    label_column_name = 'Units sold per Part'
    
    # extract the features, target and sample weight arrays
    y = dataframe[label_column_name].values
    X = dataframe.drop([label_column_name], axis=1)
    sample_weights = None
    X, y, sample_weights = data_cleaning._remove_nan_rows_in_X_y(X, y, sample_weights,
     is_timeseries=True, target_column=label_column_name)
    
    return X, y, sample_weights


def generate_data_transformation_config():
    from azureml.training.tabular.featurization._featurization_config import FeaturizationConfig
    from azureml.training.tabular.featurization.timeseries.category_binarizer import CategoryBinarizer
    from azureml.training.tabular.featurization.timeseries.max_horizon_featurizer import MaxHorizonFeaturizer
    from azureml.training.tabular.featurization.timeseries.missingdummies_transformer import MissingDummiesTransformer
    from azureml.training.tabular.featurization.timeseries.numericalize_transformer import NumericalizeTransformer
    from azureml.training.tabular.featurization.timeseries.restore_dtypes_transformer import RestoreDtypesTransformer
    from azureml.training.tabular.featurization.timeseries.rolling_window import RollingWindow
    from azureml.training.tabular.featurization.timeseries.short_grain_dropper import ShortGrainDropper
    from azureml.training.tabular.featurization.timeseries.stl_featurizer import STLFeaturizer
    from azureml.training.tabular.featurization.timeseries.time_index_featurizer import TimeIndexFeaturizer
    from azureml.training.tabular.featurization.timeseries.time_series_imputer import TimeSeriesImputer
    from azureml.training.tabular.featurization.timeseries.timeseries_transformer import TimeSeriesPipelineType
    from azureml.training.tabular.featurization.timeseries.timeseries_transformer import TimeSeriesTransformer
    from collections import OrderedDict
    from numpy import dtype
    from numpy import nan
    from sklearn.pipeline import Pipeline
    
    transformer_list = []
    transformer1 = MissingDummiesTransformer(
        numerical_columns=[]
    )
    transformer_list.append(('make_numeric_na_dummies', transformer1))
    
    transformer2 = TimeSeriesImputer(
        end=None,
        freq='MS',
        impute_by_horizon=False,
        input_column=[],
        limit=None,
        limit_direction='forward',
        method=OrderedDict([('ffill', [])]),
        option='fillna',
        order=None,
        origin=None,
        value={}
    )
    transformer_list.append(('impute_na_numeric_datetime', transformer2))
    
    transformer3 = ShortGrainDropper()
    transformer_list.append(('grain_dropper', transformer3))
    
    transformer4 = RestoreDtypesTransformer(
        dtypes={'_automl_target_col': dtype('float64')},
        target_column='_automl_target_col'
    )
    transformer_list.append(('restore_dtypes_transform', transformer4))
    
    transformer5 = STLFeaturizer(
        freq=None,
        seasonal_feature_only=False,
        seasonality=12
    )
    transformer_list.append(('make_seasonality_and_trend', transformer5))
    
    transformer6 = MaxHorizonFeaturizer(
        freq=None,
        horizon_colname='horizon_origin',
        max_horizon=12,
        origin_time_colname='origin'
    )
    transformer_list.append(('max_horizon_featurizer', transformer6))
    
    transformer7 = RollingWindow(
        backfill_cache=False,
        check_max_horizon=False,
        dropna=False,
        freq='MS',
        max_horizon=12,
        origin_time_column_name='origin',
        transform_dictionary={'min': '_automl_target_col', 'max': '_automl_target_col', 'mean': '_automl_target_col'},
        transform_options={},
        window_options={'center': False, 'closed': None},
        window_size=9
    )
    transformer_list.append(('rolling_window_operator', transformer7))
    
    transformer8 = NumericalizeTransformer(
        categories_by_col={},
        exclude_columns=set(),
        include_columns=set()
    )
    transformer_list.append(('make_categoricals_numeric', transformer8))
    
    transformer9 = TimeIndexFeaturizer(
        correlation_cutoff=0.99,
        country_or_region='US',
        datetime_columns=None,
        force_feature_list=None,
        freq='MS',
        holiday_end_time=None,
        holiday_start_time=None,
        overwrite_columns=True,
        prune_features=True
    )
    transformer_list.append(('make_time_index_featuers', transformer9))
    
    transformer10 = CategoryBinarizer(
        columns=[],
        drop_first=False,
        dummy_na=False,
        encode_all_categoricals=False,
        prefix=None,
        prefix_sep='_'
    )
    transformer_list.append(('make_categoricals_onehot', transformer10))
    
    pipeline = Pipeline(steps=transformer_list)
    tst = TimeSeriesTransformer(
        country_or_region='US',
        drop_column_names=[],
        featurization_config=FeaturizationConfig(
            blocked_transformers=None,
            column_purposes=None,
            dataset_language=None,
            prediction_transform_type=None,
            transformer_params=None
        ),
        force_time_index_features=None,
        freq='MS',
        grain_column_names=None,
        group=None,
        lookback_features_removed=False,
        max_horizon=12,
        origin_time_colname='origin',
        pipeline=pipeline,
        pipeline_type=TimeSeriesPipelineType.FULL,
        seasonality=12,
        time_column_name='Month-Year',
        time_index_non_holiday_features=['_automl_year', '_automl_year_iso', '_automl_half', '_automl_quarter', '_automl_month', '_automl_wday', '_automl_qday', '_automl_week'],
        use_stl='season_trend'
    )
    
    return tst
    
    
def generate_preprocessor_config():
    from sklearn.preprocessing import StandardScaler
    
    preproc = StandardScaler(
        copy=True,
        with_mean=False,
        with_std=False
    )
    
    return preproc
    
    
def generate_algorithm_config():
    from xgboost.sklearn import XGBRegressor
    
    algorithm = XGBRegressor(
        base_score=0.5,
        booster='gbtree',
        colsample_bylevel=1,
        colsample_bynode=1,
        colsample_bytree=0.8,
        eta=0.05,
        gamma=0.01,
        gpu_id=-1,
        importance_type='gain',
        interaction_constraints='',
        learning_rate=0.0500000007,
        max_delta_step=0,
        max_depth=10,
        max_leaves=15,
        min_child_weight=1,
        missing=numpy.nan,
        monotone_constraints='()',
        n_estimators=25,
        n_jobs=0,
        num_parallel_tree=1,
        objective='reg:squarederror',
        random_state=0,
        reg_alpha=1.1458333333333335,
        reg_lambda=0.20833333333333334,
        scale_pos_weight=1,
        subsample=1,
        tree_method='auto',
        validate_parameters=1,
        verbose=-10,
        verbosity=0
    )
    
    return algorithm
    
    
def build_model_pipeline():
    from azureml.training.tabular.models.forecasting_pipeline_wrapper import ForecastingPipelineWrapper
    from sklearn.pipeline import Pipeline
    
    logger.info("Running build_model_pipeline")
    pipeline = Pipeline(
        steps=[
            ('tst', generate_data_transformation_config()),
            ('model', generate_algorithm_config())
        ]
    )
    forecast_pipeline_wrapper = ForecastingPipelineWrapper(pipeline, stddev=[12.847395252041215, 12.696178222300148, 6.929608582524011, 3.438912429865157, 3.616586289126778, 15.568061428307269, 14.220140547381748, 16.380051396919406, 19.966958022800963, 18.414634577096415, 11.71739856092113, 12.597716342750218])
    
    return forecast_pipeline_wrapper
def get_metrics_names():
    regression_metrics_names = [
        'root_mean_squared_error',
        'explained_variance',
        'mean_absolute_percentage_error',
        'r2_score',
        'root_mean_squared_log_error',
        'mean_absolute_error',
        'spearman_correlation',
        'residuals',
        'median_absolute_error',
        'predicted_true',
    ]
    forecasting_metrics_names = [
        'forecast_mean_absolute_percentage_error',
        'forecast_table',
        'forecast_residuals',
    ]
    return regression_metrics_names, forecasting_metrics_names


def calculate_metrics(model, X, y, sample_weights, X_test, y_test, cv_splits=None):
    from azureml.training.tabular.preprocessing.binning import get_dataset_bins
    from azureml.training.tabular.score.scoring import score_forecasting
    from azureml.training.tabular.score.scoring import score_regression
    
    y_pred, _ = model.forecast(X_test)
    y_min = np.min(y)
    y_max = np.max(y)
    y_std = np.std(y)
    
    bin_info = get_dataset_bins(cv_splits, X, None, y)
    regression_metrics_names, forecasting_metrics_names = get_metrics_names()
    metrics = score_regression(
        y_test, y_pred, regression_metrics_names, y_max, y_min, y_std, sample_weights, bin_info)
    
    try:
        horizons = X_test['horizon_origin'].values
    except Exception:
        # If no horizon is present we are doing a basic forecast.
        # The model's error estimation will be based on the overall
        # stddev of the errors, multiplied by a factor of the horizon.
        horizons = np.repeat(None, y_pred.shape[0])
    
    featurization_step = generate_data_transformation_config()
    grain_column_names = featurization_step.grain_column_names
    time_column_name = featurization_step.time_column_name
    
    forecasting_metrics = score_forecasting(
        y_test, y_pred, forecasting_metrics_names, horizons, y_max, y_min, y_std, sample_weights, bin_info,
        X_test, X, y, grain_column_names, time_column_name)
    metrics.update(forecasting_metrics)
    return metrics


def train_model(X, y, sample_weights=None, transformer=None):
    logger.info("Running train_model")
    model_pipeline = build_model_pipeline()
    
    model = model_pipeline.fit(X, y)
    return model


def main(training_dataset_id=None):
    from azureml.core.run import Run
    from azureml.training.tabular.score._cv_splits import _CVSplits
    from azureml.training.tabular.score.scoring import aggregate_scores
    
    # The following code is for when running this code as part of an AzureML script run.
    run = Run.get_context()
    
    df = get_training_dataset(training_dataset_id)
    print(df.head())
    X, y, sample_weights = prepare_data(df)
    tst = generate_data_transformation_config()
    tst.fit(X, y)
    ts_param_dict = tst.parameters
    short_series_dropper = next((step for key, step in tst.pipeline.steps if key == 'grain_dropper'), None)
    if short_series_dropper is not None and short_series_dropper.has_short_grains_in_train and grains is not None and len(grains) > 0:
        # Preprocess X so that it will not contain the short grains.
        dfs = []
        X['_automl_target_col'] = y
        for grain, df in X.groupby(grains):
            if grain in short_series_processor.grains_to_keep:
                dfs.append(df)
        X = pd.concat(dfs)
        y = X.pop('_automl_target_col').values
        del dfs
    cv_splits = _CVSplits(X, y, frac_valid=None, CV=5, n_step=None, is_time_series=True, task='regression', timeseries_param_dict=ts_param_dict)
    scores = []
    for X_train, y_train, sample_weights_train, X_valid, y_valid, sample_weights_valid in cv_splits.apply_CV_splits(X, y, sample_weights):
        partially_fitted_model = train_model(X_train, y_train, transformer=tst)
        metrics = calculate_metrics(partially_fitted_model, X, y, sample_weights, X_test=X_valid, y_test=y_valid, cv_splits=cv_splits)
        scores.append(metrics)
        print(metrics)
    model = train_model(X_train, y_train, transformer=tst)
    
    metrics = aggregate_scores(scores)
    
    print(metrics)
    for metric in metrics:
        run.log(metric, metrics[metric])
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    run.upload_file('outputs/model.pkl', 'model.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_dataset_id', type=str, default='3393406d-7dc3-4adb-b7ce-282a7ecf47be', help='Default training dataset id is populated from the parent run')
    args = parser.parse_args()
    
    main(args.training_dataset_id)