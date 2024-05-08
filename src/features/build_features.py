import sys
import pandas as pd
import numpy as np
from pathlib import Path
from distances import haversine_distance, euclidean_distance, manhattan_distance

new_feature_names = ['haversine_distance',
                     'euclidean_distance',
                     'manhattan_distance']

build_features_list = [haversine_distance,
                       euclidean_distance,
                       manhattan_distance]


def implement_distances(dataframe:pd.DataFrame, 
                        lat1:pd.Series, 
                        lon1:pd.Series, 
                        lat2:pd.Series, 
                        lon2:pd.Series) -> pd.DataFrame:
    dataframe = dataframe.copy()
    for ind in range(len(build_features_list)):
        func = build_features_list[ind]
        dataframe[new_feature_names[ind]] = func(lat1,lon1,
                                                 lat2,lon2)
    
    return dataframe

def read_dataframe(path):
    df = pd.read_csv(path)
    return df

def save_dataframe(dataframe:pd.DataFrame, save_path):
    dataframe.to_csv(save_path,index=False)

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
        df = read_dataframe(data_path)
        # build features for dataframe
        df = implement_distances(dataframe=df,
                                 lat1=df['pickup_latitude'],
                                 lon1=df['pickup_longitude'],
                                 lat2=df['dropoff_latitude'],
                                 lon2=df['dropoff_longitude'])
        # save the dataframe
        output_path = root_path / "data/processed/build-features"
        # make the directory if not available
        output_path.mkdir(parents=True,exist_ok=True)
        # save the data
        save_dataframe(df,output_path / filename)