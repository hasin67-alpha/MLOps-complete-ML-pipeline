

import logging
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder



log_dirs="logs"
os.makedirs(log_dirs,exist_ok=True)

# logger 
logger=logging.getLogger('Data Preprocessing')
logger.setLevel('DEBUG')


# console handler for show result 
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

# file handler for show log file 

log_file_path=os.path.join(log_dirs,"data_preprocessing.log")

file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


logger.addHandler(console_handler)
logger.addHandler(file_handler)




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






def preprocessing_df(df:pd.DataFrame):
    try:
        le = LabelEncoder()
        df['Type'] = le.fit_transform(df['Type'])
        logger.debug('Type column encoded')
        return df
    
    except KeyError as e:
        logger.error('Column not found: %s', e)
        raise

    except Exception as e:
        logger.error('Error during label encoding process: %s', e)
        raise




def main():

    try: 
        train_data=load_data('./data/raw/train.csv')
        test_data=load_data('./data/raw/test.csv')
        logger.debug('Data loaded properly')
        
        # encoding 
        train_processed_data=preprocessing_df(df=train_data)
        test_processed_data=preprocessing_df(df=test_data)

        # Store the data inside data/processed
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
            
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        
        logger.debug('Processed data saved to %s', data_path)
    

    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")





if __name__ == '__main__':
    main()









