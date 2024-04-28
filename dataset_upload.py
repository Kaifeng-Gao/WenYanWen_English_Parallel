from datasets import load_from_disk, load_dataset
from datasets import concatenate_datasets
import yaml

CONFIG_PATH = 'config.yaml'
DATASET_PATH = "KaifengGGG/WenYanWen_English_Parrallel"

def load_config(config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
config = load_config(CONFIG_PATH)
token = config['access_token']['huggingface_token']

ds = []
for i in range(1,11):
    path = f"datasets/dataset_instruct_message/subset_{i}"
    ds.append(load_from_disk(path))

dataset_instruct = concatenate_datasets(ds)
dataset_final = dataset_instruct.train_test_split(test_size=0.1, shuffle=False)
dataset_final

dataset_final.push_to_hub(DATASET_PATH, 'instruct-large', token=token)