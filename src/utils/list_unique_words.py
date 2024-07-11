import json

def list_unique_words(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    unique_words = set()
    for entry in data:
        instruction = entry['instruction']
        words = instruction.split()
        unique_words.update(words)
    return unique_words

training_words = list_unique_words('data/training_examples.json')
testing_words = list_unique_words('data/testing_examples.json')

all_words = training_words.union(testing_words)
print("Unique words in dataset:", all_words)
