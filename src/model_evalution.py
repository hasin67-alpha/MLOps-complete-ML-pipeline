import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import logging
import yaml
from dvclive import Live



log_dirs="logs"
os.makedirs(log_dirs,exist_ok=True)

# logger 
logger=logging.getLogger('model_evalution')
logger.setLevel('DEBUG')


# console handler for show result 
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

# file handler for show log file 

log_file_path=os.path.join(log_dirs,"model_evalution.log")

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













def load_model(file_path: str):
   
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model: %s', e)
        raise





def load_data(file_path: str) -> pd.DataFrame:
   
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise




def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:

    try:
        params=load_params(params_path='params.yaml')

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        # logger.debug(f" Accuracy: {acc:.4f}")

        conf_matrix=confusion_matrix(y_test, y_pred)
        # logger.debug(f"Confusion Matrix: \n {conf_matrix}")

        cls_report=classification_report(y_test, y_pred, output_dict=True)
        # logger.debug(f"Classification Report: \n {cls_report}")

        metrics_summary = {
            "accuracy": float(acc),
            "confusion_matrix": conf_matrix.tolist(), 
            "classification_report": cls_report
        }



        # real cls name find from y_test
        class_names = [str(c) for c in np.unique(y_test)]
        class_a = class_names[0]
        class_b = class_names[1] 

        # dvclive implement for save params when checking different numbers
        with Live(save_dvc_exp=True) as live:
            
            # acc record
            live.log_metric('accuracy', float(acc))
    
            #  class-1 (precision and recall)
            live.log_metric(f'precision_{class_a}', cls_report[class_a]['precision'])
            live.log_metric(f'recall_{class_a}', cls_report[class_a]['recall'])
            
            #  class-2 (precision and recall)
            live.log_metric(f'precision_{class_b}', cls_report[class_b]['precision'])
            live.log_metric(f'recall_{class_b}', cls_report[class_b]['recall'])
            
            # overall 'macro avg' Precision ও Recall লগ করা
            live.log_metric('macro_precision', cls_report['macro avg']['precision'])
            live.log_metric('macro_recall', cls_report['macro avg']['recall'])
            
            # params save 
            live.log_params(params)
             
        return metrics_summary

    except ValueError as val_err:
        logger.error(f"Data format or dimension mismatch error: {str(val_err)}")
        raise val_err
        
    except AttributeError as attr_err:
        logger.error(f"The provided model object is invalid or not trained: {str(attr_err)}")
        raise attr_err
        
    except Exception as e:
        logger.error(f"Unexpected error occurred during model evaluation: {str(e)}")
        raise e
    


def save_metrics(metrics: dict, file_path: str) -> None:
    
    try:
       
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise




def main():
    try:
        model=load_model(file_path="./model/model.pkl")
        test_data=load_data(file_path="./data/interim/test_processed.csv")
        
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics=evaluate_model(model,X_test,y_test)


        save_metrics(metrics,'reports/metrics.json')

    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")







if __name__ == '__main__':
    main()


