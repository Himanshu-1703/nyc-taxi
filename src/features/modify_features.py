import sys
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from src.logger import CustomLogger, create_log_path


TARGET_COLUMN = 'trip_duration'
PLOT_PATH = Path("reports/figures/target_distribution.png")

## Logging
# set the logger path
log_file_path = create_log_path('modify_features')
# create the custom logger object
modify_logger = CustomLogger(logger_name='modify_features',
                              log_filename=log_file_path)
# set the level of logging to INFO
modify_logger.set_log_level(level=logging.INFO)

## Functions applied on target column
def convert_target_to_minutes(dataframe: pd.DataFrame, target_column: str) -> pd.DataFrame:
    # convert the target into minutes
    dataframe.loc[:,target_column] = dataframe[target_column] / 60
    modify_logger.save_logs(msg='Target column converted from seconds into minutes')
    return dataframe

def drop_above_two_hundred_minutes(dataframe: pd.DataFrame, target_column: str) -> pd.DataFrame:
    # filter rows where target is less than 200
    filter_series = dataframe[target_column] <= 200
    new_dataframe = dataframe.loc[filter_series,:].copy()
    # max value of target column to checjk the outliers are removed
    max_value = new_dataframe[target_column].max()
    modify_logger.save_logs(msg=f'The max value in target column after transformation is {max_value} and the state of transformation is {max_value <= 200}')
    if max_value <= 200:
        return new_dataframe
    else:
        raise ValueError('Outlier target values not removed from the data')        


def plot_target(dataframe: pd.DataFrame, target_column: str, save_path: str):
    # plot the density plot of the target after transformation
    sns.kdeplot(data=dataframe, x=target_column)
    plt.title(f'Distribution of {target_column}')
    # save the plot at the destination path
    plt.savefig(save_path)
    modify_logger.save_logs(msg='Distribution plot saved at destination')
    
    
def drop_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    modify_logger.save_logs(f'Columns in data before removal are {list(dataframe.columns)}')
    # drop columns from train and val data
    if 'dropoff_datetime' in dataframe.columns:
        columns_to_drop = ['id','dropoff_datetime','store_and_fwd_flag']
        # dropping the columns from dataframe
        dataframe_after_removal = dataframe.drop(columns=columns_to_drop)
        list_of_columns_after_removal = list(dataframe_after_removal.columns)
        modify_logger.save_logs(f'Columns in data after removal are {list_of_columns_after_removal}')
        # verifying if columns dropped
        modify_logger.save_logs(msg=f"Columns {', '.join(columns_to_drop)} dropped from data  verify={columns_to_drop not in list_of_columns_after_removal}")
        return dataframe_after_removal
    # drop columns from the test data
    else:
        columns_to_drop = ['id','store_and_fwd_flag']
        # dropping the columns from dataframe
        dataframe_after_removal = dataframe.drop(columns=columns_to_drop)
        list_of_columns_after_removal = list(dataframe_after_removal.columns)
        # verifying if columns dropped
        modify_logger.save_logs(msg=f"Columns {', '.join(columns_to_drop)} dropped from data  verify={columns_to_drop not in list_of_columns_after_removal}")
        return dataframe_after_removal


def make_datetime_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    # copy the original dataframe
    new_dataframe = dataframe.copy()
    # number of rows and column before transformation
    original_number_of_rows, original_number_of_columns = new_dataframe.shape
    
    # convert the column to datetime column
    new_dataframe['pickup_datetime'] = pd.to_datetime(new_dataframe['pickup_datetime'])
    modify_logger.save_logs(msg=f'pickup_datetime column converted to datetime {new_dataframe["pickup_datetime"].dtype}')
    
    # do the modifications
    new_dataframe.loc[:,'pickup_hour'] = new_dataframe['pickup_datetime'].dt.hour 
    new_dataframe.loc[:,'pickup_date'] = new_dataframe['pickup_datetime'].dt.day
    new_dataframe.loc[:,'pickup_month'] = new_dataframe['pickup_datetime'].dt.month
    new_dataframe.loc[:,'pickup_day'] = new_dataframe['pickup_datetime'].dt.weekday
    new_dataframe.loc[:,'is_weekend'] = new_dataframe.apply(lambda row: row['pickup_day'] >= 5,axis=1).astype('int')
    
    # drop the redundant date time column
    new_dataframe = new_dataframe.drop(columns=['pickup_datetime'])
    modify_logger.save_logs(msg=f'pickup_datetime column dropped  verify={"pickup_datetime" not in new_dataframe.columns}')
    
    # number of rows and columns after transformation
    transformed_number_of_rows, transformed_number_of_columns = new_dataframe.shape
    modify_logger.save_logs(msg=f'The number of columns increased by 4 {transformed_number_of_columns == (original_number_of_columns + 5 - 1)}')
    modify_logger.save_logs(msg=f'The number of rows remained the same {original_number_of_rows == transformed_number_of_rows}')
    return new_dataframe


def remove_passengers(dataframe: pd.DataFrame) -> pd.DataFrame:
    # make the list of passenger to keep
    passengers_to_include = list(range(1,7))
    # filter out rows which matches exavctly the passengers in the list
    new_dataframe_filter = dataframe['passenger_count'].isin(passengers_to_include)
    # filter the dataframe
    new_dataframe = dataframe.loc[new_dataframe_filter,:]
    # list of unique passenger values in the passenger_count column
    unique_passenger_values = list(np.sort(new_dataframe['passenger_count'].unique()))
    modify_logger.save_logs(msg=f'The unique passenger list is {unique_passenger_values}  verify={passengers_to_include == unique_passenger_values}')
    return new_dataframe


def input_modifications(dataframe: pd.DataFrame) -> pd.DataFrame:
    # drop the columns in input data
    new_df = drop_columns(dataframe)
    # remove the rows having excluded passenger values
    df_passengers_modifications = remove_passengers(new_df)
    # add datetime features to data
    df_with_datetime_features = make_datetime_features(df_passengers_modifications)
    modify_logger.save_logs('Modifications with input features complete')
    return df_with_datetime_features

   
def target_modifications(dataframe: pd.DataFrame, target_column: str=TARGET_COLUMN) -> pd.DataFrame:
    # convert the target column from seconds to minutes
    minutes_dataframe = convert_target_to_minutes(dataframe,target_column)
    # remove target values greater than 200
    target_outliers_removed_df = drop_above_two_hundred_minutes(minutes_dataframe,target_column)
    # plot the target column
    plot_target(dataframe=target_outliers_removed_df,target_column=target_column,
                save_path=root_path / PLOT_PATH)
    modify_logger.save_logs('Modifications with the target feature complete')
    return target_outliers_removed_df

# read the dataframe from location
def read_data(data_path):
    df = pd.read_csv(data_path)
    return df

# save the dataframe to location
def save_data(dataframe: pd.DataFrame,save_path: Path):
    dataframe.to_csv(save_path,index=False)
    
    
# TODO 1. Make a function to read the dataframe from the dvc.yaml file
# TODO 2. Add Logging Functionality to each function
# TODO 3. Run the code in notebook mode to test with print statements
# ? Should logging be added to each function or the main function for specific steps


def main(data_path,filename):
    # read the data into dataframe
    df = read_data(data_path)
    # do the modifications on the input data
    df_input_modifications = input_modifications(dataframe=df)
    # check whether the input file has target column
    if (filename == "train.csv") or (filename == "val.csv"):
        df_final = target_modifications(dataframe=df_input_modifications)  
    else:
        df_final = df_input_modifications
        
    return df_final
        

if __name__ == "__main__":
    for ind in range(1,4):
        # read the input file name from command
        input_file_path = sys.argv[ind]
        # current file path
        current_path = Path(__file__)
        # root directory path
        root_path = current_path.parent.parent.parent
        # input data path
        data_path = root_path / input_file_path
        # get the file name
        filename = data_path.parts[-1]
        # call the main function
        df_final = main(data_path=data_path,filename=filename)
        # save the dataframe
        output_path = root_path / "data/processed/transformations"
        # make the directory if not available
        output_path.mkdir(parents=True,exist_ok=True)
        # save the data
        save_data(df_final,output_path / filename)
        modify_logger.save_logs(msg=f'{filename} saved at the destination folder')