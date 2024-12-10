import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.get_data import get_data_path
import kagglehub

def main():
    dataset_name = "bhadramohit/customer-shopping-latest-trends-dataset"
    dataset_path = get_data_path()
    
    path = kagglehub.dataset_download(dataset_name)
    
    dataset_file = os.path.join(path, 'customer-shopping-latest-trends-dataset.csv')
    
    if os.path.exists(dataset_file):
        df_customer_shopping = pd.read_csv(dataset_file)
        print(df_customer_shopping.head())
    else:
        print(f"Datasetbestand niet gevonden op: {dataset_file}")

if __name__ == "__main__":
    main()