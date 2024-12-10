import kagglehub
import os

from .get_data import get_data_path

dataset_name = "bhadramohit/customer-shopping-latest-trends-dataset"

data_path = get_data_path()

path = kagglehub.dataset_download(dataset_name)

print(f"Path to dataset files: {path}")