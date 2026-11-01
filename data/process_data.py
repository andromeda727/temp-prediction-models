import pandas as pd
import sklearn.preprocessing as pp
from sklearn.model_selection import train_test_split
from pathlib import Path

def create_processed_data_csv(path_to_raw: Path, path_to_processed: Path, overwrite: bool = False) -> bool:
    '''function for processing raw Jena climate data csv into usable data for our ML problem
    Parameters:
    path_to_raw: pathlib.Path -- path to raw csv file containing Jena climate data (sourced from Kaggle: https://www.kaggle.com/datasets/mnassrib/jena-climate)
    path_to_processed: pathlib.Path -- output path for processed data csv when completed
    
    Returns:
    bool: whether data processing was successful (True) or unsuccessful (False)'''
    
    if path_to_processed.is_file() and not overwrite:
        print(f'Processed Data CSV at specified path {path_to_processed} already exists. Either specify overwrite=true, or choose a different output path\n')
        return False
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
    new_labels = pd.date_range(start=features_datetime_indexed.index.min().floor('D'), 
                  end=features_datetime_indexed.index.max().ceil('D') - pd.Timedelta(minutes=10),
                  freq='10min') 
    # create a full index list from beginning of first day of dataset to end of last day of dataset (in case any times are missing in the dataset).
    # we use this to reindex our dataset and insert NaNs wherever missing times/rows exist
    data_reindexed = features_datetime_indexed.reindex(new_labels)
    data_np = data_reindexed.to_numpy()
    

    return True


    
    