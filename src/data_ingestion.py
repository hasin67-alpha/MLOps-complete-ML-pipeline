
import pandas as pd
import os
import logging
from sklearn.model_selection import train_test_split



# log file create 
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)


# logging module add 
logger=logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

# data show in console 
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

# data show in log file 
log_file_path=os.path.join(log_dir,"data_ingestion.log")
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# formate the log output 
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


# logger object connect with file and console handler for pass the data 
logger.addHandler(file_handler)
logger.addHandler(console_handler)




def load_data(data_url:str)->pd.DataFrame:
    
    try:
        df=pd.read_csv(data_url)
        logger.debug("Data Loaded from %s",data_url)
        return df

    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s',e)
        raise

    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def preprocess_data(df:pd.DataFrame)->pd.DataFrame:
    try:
        df.drop(columns=['UDI','Product ID','TWF','HDF','PWF','OSF','RNF'], inplace=True)
        df.rename(columns={'Air temperature [K]':'Air temperature', 'Process temperature [K]':'Process temperature', 'Rotational speed [rpm]' : 'Rotational speed', 'Torque [Nm]':'Torque', 'Tool wear [min]':'Tool wear'},inplace=True)
        logger.debug('Data Preporocessing completed')
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise

    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise


def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str)-> None:
    try:
        raw_data_path=os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug('Train and test data saved to %s', raw_data_path)

    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise



def main():

    try:
        test_size=0.2
        data_path="experiments/predictive_maintenance.csv"
        df=load_data(data_url=data_path)
        final_df=preprocess_data(df)
        train_data,test_data=train_test_split(final_df,test_size=test_size,random_state=2,stratify=final_df['Machine failure'])
        save_data(train_data,test_data,data_path="./data")
        # print(train_data.shape)
        # print(test_data.shape)
    
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")


if __name__=='__main__':
    main()










