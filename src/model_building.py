
import logging
import os
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split,GridSearchCV
from xgboost import XGBClassifier

import xgboost as xgb
import yaml



log_dirs="logs"
os.makedirs(log_dirs,exist_ok=True)

# logger 
logger=logging.getLogger('model_building')
logger.setLevel('DEBUG')


# console handler for show result 
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

# file handler for show log file 

log_file_path=os.path.join(log_dirs,"model_building.log")

file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


logger.addHandler(console_handler)
logger.addHandler(file_handler)







def load_params(params_path:str)->dict:
    try:
        with open(params_path,'r') as file:
            params=yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise



def load_data(file_path: str) -> pd.DataFrame:
    
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded  %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise




def train_model(df:pd.DataFrame):

    try:

        params=load_params(params_path='params.yaml')

        X=df.drop(columns='Machine failure')
        y=df['Machine failure']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        

        negative_cases = np.sum(y_train == 0)
        positive_cases = np.sum(y_train == 1)
        weight = negative_cases / positive_cases

        logger.debug("Calculated scale_pos_weight")
        

        # Base Model
        base_model = xgb.XGBClassifier(scale_pos_weight=weight, random_state=42, eval_metric='logloss')

        # parameter tuining
        param_grid =params['params_grid']

        # GridSearchCV
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring=params['model_building']['scoring'],
            cv=params['model_building']['cv_folds'],
            verbose=1,
            n_jobs=-1
        )

        # model training
        grid_search.fit(X_train, y_train)

        logger.debug("Training completed")

        logger.debug(f"best parameter: {grid_search.best_params_}\n")

        return grid_search.best_estimator_
        
    except Exception as e:

        logger.error(f"Unexpected error occurred during model training: {str(e)}")
        raise e



def save_model(model):

    try:
        os.makedirs("model", exist_ok=True)
        with open("model/model.pkl", "wb") as file:
            pickle.dump(model, file)
            logger.debug("Model saved duccessfully")
    
    except FileNotFoundError as e:
        logger.error('File path not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:
        train_data = load_data('./data/interim/train_processed.csv')

        model=train_model(train_data)
        save_model(model=model)

    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

        




if __name__ == '__main__':
    main()






