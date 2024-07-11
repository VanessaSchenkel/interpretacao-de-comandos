import json
import os

class Dataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def _load_data(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return data

    def get_train_data(self):
        train_data = self._load_data(os.path.join(self.data_dir, 'train.json'))
        print(f"Loaded {len(train_data)} training examples")
        return train_data

    def get_test_data(self):
        test_data = self._load_data(os.path.join(self.data_dir, 'test.json'))
        print(f"Loaded {len(test_data)} testing examples")
        return test_data
