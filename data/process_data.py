import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import einops
import numpy as np


def create_processed_data_csv(path_to_raw: Path, path_to_processed: Path, overwrite: bool = False) -> tuple[bool, StandardScaler | None]:
    '''function for processing raw Jena climate data csv into usable data for our ML problem
    Parameters:
    path_to_raw: pathlib.Path -- path to raw csv file containing Jena climate data (sourced from Kaggle: https://www.kaggle.com/datasets/mnassrib/jena-climate)
    path_to_processed: pathlib.Path -- output path for processed data storage when completed
    
    Returns:
    bool: whether data processing was successful (True) or unsuccessful (False)
    StandardScaler: scaler used for normalizing features'''
    
    if path_to_processed.is_file() and not overwrite:
        print(f'Processed Data at specified path {path_to_processed} already exists. Either specify overwrite=true, or choose a different output path\n')
        return False, None
    if not path_to_raw.is_file():
        raise ValueError(f'no file exists at the specified path to raw data: {path_to_raw}')

    data_raw = pd.read_csv(path_to_raw)
    print(data_raw.head())

    FEATURES = [
        'Date Time',
        'p (mbar)',
        'T (degC)',
        'wv (m/s)',
        'wd (deg)',
        'rho (g/m**3)'
    ]

    features_of_interest = data_raw[FEATURES]
    datetime_formatted = pd.to_datetime(features_of_interest['Date Time'], format='%d.%m.%Y %H:%M:%S')
    features_of_interest['Date Time'] = datetime_formatted
    features_datetime_indexed = features_of_interest.sort_values('Date Time').set_index('Date Time') # sort by datetime to make sure, then set datetimes as index label
    new_idx_labels = pd.date_range(start=features_datetime_indexed.index.min().floor('D'), 
                  end=features_datetime_indexed.index.max().ceil('D') - pd.Timedelta(minutes=10),
                  freq='10min') 
    # create a full index list from beginning of first day of dataset to end of last day of dataset (in case any times are missing in the dataset).
    # we use this to reindex our dataset and insert NaNs wherever missing times/rows exist
    data_reindexed = features_datetime_indexed.reindex(new_idx_labels)
    data_np = data_reindexed.to_numpy()
    num_examples_per_day = 144
    # d = day, e = examples per day, f = features
    grouped_data = einops.rearrange(data_np, '(d e) f -> d e f', e=num_examples_per_day)

    # creating labels to train against (min temp & max temp of next day)
    temp_col = 1
    temps = grouped_data[:, :, temp_col] # shape = num_days, 144
    day_maxes = np.nanmax(temps, axis=1) # shape = num_days,
    day_mins = np.nanmin(temps, axis=1) # shape = num_days,
    y_ = np.hstack([day_mins[1:], day_maxes[1:]]) # shape = num_days - 1, 2
    X_ = grouped_data[:-1, :, :] # shape = num_days - 1, 144, num_features
    # since our prediction problem is predicting next-day temps, we need to drop the last day from our feature dataset, and drop the first day's temps from our label dataset


    # split into train, validation, and test data (80% train, 10% validation, 10% test)
    X_tr, X_test, y_tr, y_test = train_test_split(X_, y_, test_size=0.1, shuffle=False)
    val_ratio = 0.1 / 0.9
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=val_ratio, shuffle=False)

    # drop last examples in each (leakage from train --> val --> test)
    X_tr, y_tr, X_val, y_val = X_tr[:-1], y_tr[:-1], X_val[:-1], y_val[:-1]

    # standardizing features
    scaler = StandardScaler()

    X_tr_scaling, X_val_scaling, X_test_scaling = einops.rearrange(X_tr, 'd e f -> (d e) f'), einops.rearrange(X_val, 'd e f -> (d e) f'), einops.rearrange(X_test, 'd e f -> (d e) f')
   
    X_tr_scaled = einops.rearrange(scaler.fit_transform(X_tr_scaling), '(d e) f -> d e f') # compute and store scaled X_tr
    X_val_scaled = einops.rearrange(scaler.fit_transform(X_val_scaling), '(d e) f -> d e f') # compute and store scaled X_val
    X_test_scaled = einops.rearrange(scaler.fit_transform(X_test_scaling), '(d e) f -> d e f') # compute and store scaled X_test

    # create a mask for NaNs and then replace them so that the replaced values don't affect our models/predictions (as much)
    def fill_nans_and_stack_mask(X: np.ndarray) -> np.ndarray:
        '''helper function for creating a NaN mask on X, filling X's NaNs, and fusing/stacking the mask tensor and X tensor together'''
        nan_mask = np.isnan(X) # shape = n_days, 144, n_features (preserved from X)
        mask = ~nan_mask # invert NaN mask to tell us (True) where there was originally data, and (False) where no data (NaN) was present
        filled = np.nan_to_num(X, nan=0.0) # works best if mean of X is 0 (X is z-score normalized), shape = n_days, 144, n_features
        return np.concatenate([filled, mask.astype(np.float32)], axis=2) # shape = n_days, 144, 2*num_features -- note that we convert the True/False mask into 1s and 0s as well

    X_tr_final = fill_nans_and_stack_mask(X_tr_scaled)
    X_val_final = fill_nans_and_stack_mask(X_val_scaled)
    X_test_final = fill_nans_and_stack_mask(X_test_scaled)

    np.savez(path_to_processed.with_suffix('.npz'),
         X_train=X_tr_final,
         X_validate=X_val_final,
         X_test=X_test_final,
         y_train=y_tr,
         y_val=y_val,
         y_test=y_test)
    
    return True, scaler


    
    