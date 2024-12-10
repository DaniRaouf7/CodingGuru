import os
import pandas as pd
import kagglehub  # type: ignore


def main():
    dataset_name = "bhadramohit/customer-shopping-latest-trends-dataset"

    path = kagglehub.dataset_download(dataset_name)
    dataset_file = os.path.join(path, 'shopping_trends.csv')

    if os.path.exists(dataset_file):
        df_customer_shopping = pd.read_csv(dataset_file)
        print(df_customer_shopping.head())
    else:
        print(f"Datasetbestand niet gevonden op: {dataset_file}")


if __name__ == "__main__":
    main()
