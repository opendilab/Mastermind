import os
import pandas as pd


def load_dataset(root_path, split):
    dataset_path = os.path.join(f'hf://datasets/OpenDILabCommunity/MasterMind/{root_path}/', f'{split}_dataset.parquet')
    print(dataset_path)
    df = pd.read_parquet(dataset_path)
    return df.to_dict('records')

