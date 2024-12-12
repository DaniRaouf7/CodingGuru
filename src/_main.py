import os
import pandas as pd
import kagglehub  # type: ignore

from src.data_preprocessing import preprocess_data
from src.model_training import train_and_optimize_model_rf


def main():
    dataset_name = "bhadramohit/customer-shopping-latest-trends-dataset"

    path = kagglehub.dataset_download(dataset_name)
    dataset_file = os.path.join(path, 'shopping_trends.csv')

    pd.set_option('display.max_columns', None)

    if os.path.exists(dataset_file):
        df_customer_shopping = pd.read_csv(dataset_file)
        print(df_customer_shopping.head())
    else:
        print(f"Datasetbestand niet gevonden op: {dataset_file}")

    # Preprocess
    X_train, X_test, y_train, y_test = preprocess_data(dataset_file)

    # Train model
    train_and_optimize_model_rf(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
