from datasets import Dataset

def create_hf_dataset(records):
    return Dataset.from_list(records)