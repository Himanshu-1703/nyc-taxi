import joblib
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score


TARGET = 'trip_duration'
model_name = 'xgbreg.joblib'


def load_dataframe(path):
    df = pd.read_csv(path)
    return df
    
    
def make_X_y(dataframe:pd.DataFrame,target_column:str):
    df_copy = dataframe.copy()
    
    X = df_copy.drop(columns=target_column)
    y = df_copy[target_column]
    
    return X, y

def get_predictions(model,X:pd.DataFrame):
    # get predictions on data
    y_pred = model.predict(X)
    
    return y_pred
    
def calculate_r2_score(y_actual,y_predicted):
    score = r2_score(y_actual,y_predicted)
    return score


def main():
    # current file path
    current_path = Path(__file__)
    # root directory path
    root_path = current_path.parent.parent.parent
    # read input file path
    data_path = root_path / 'data/processed/final'
    for i in range(1,3):
        filename = sys.argv[i]
        # load the data 
        data = load_dataframe(data_path / filename)
        # split the data into X and y
        X, y = make_X_y(dataframe=data,target_column=TARGET)
        # model path
        model_path = root_path / 'models' / 'models' / model_name
        # load the model
        model = joblib.load(model_path)
        if filename == "train.csv":
            # get predictions from model
            cross_val = cross_val_score(estimator=model,
                                        X=X,y=y,
                                        cv=10,scoring='r2',
                                        n_jobs=-1)
            x_axis_list = [f'fold_{axis}' for axis in range(1,11)]
            y_axis_list = list(cross_val)  
        else:
            # calculate the y_pred
            y_pred = get_predictions(model=model,X=X)
            r2_score = calculate_r2_score(y,y_pred)
            x_axis_list.append('val')
            y_axis_list.append(r2_score)
            
    # results save path
    results_path = Path(root_path / "plots" / "model_results")
    results_path.mkdir(exist_ok=True)
    
    # plot the graph
    fig = plt.figure(figsize=(15,8))
    plt.bar(x=x_axis_list,height=y_axis_list)
    plt.xlabel("K folds")
    plt.ylabel("R2 Score")
    fig.savefig(results_path / "results.png")    
        
if __name__ == "__main__":
    main()