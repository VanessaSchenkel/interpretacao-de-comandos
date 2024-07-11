import json
import csv
import os

def preprocess_data():
    raw_data_dir = 'data/raw_data'
    instructions_file = os.path.join(raw_data_dir, 'instructions_pt.csv')
    initial_states_file = os.path.join(raw_data_dir, 'initial_states.json')
    final_states_file = os.path.join(raw_data_dir, 'final_states.json')

    processed_data_dir = 'data/processed_data'
    train_file = os.path.join(processed_data_dir, 'train.json')
    val_file = os.path.join(processed_data_dir, 'val_dataset.json')
    test_file = os.path.join(processed_data_dir, 'test.json')
    vocab_file = os.path.join(processed_data_dir, 'vocab.json')

    with open(instructions_file, 'r') as f:
        instructions = list(csv.DictReader(f))

    with open(initial_states_file, 'r') as f:
        initial_states = json.load(f)

    with open(final_states_file, 'r') as f:
        final_states = json.load(f)

    processed_data = []
    for instruction in instructions:
        data_entry = {
            "instruction": instruction['instruction'],
            "initial_state": initial_states[instruction['id']],
            "final_state": final_states[instruction['id']]
        }
        processed_data.append(data_entry)

    train_data = processed_data[:int(0.7 * len(processed_data))]
    val_data = processed_data[int(0.7 * len(processed_data)):int(0.85 * len(processed_data))]
    test_data = processed_data[int(0.85 * len(processed_data)):]

    os.makedirs(processed_data_dir, exist_ok=True)

    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=4)

    with open(val_file, 'w') as f:
        json.dump(val_data, f, indent=4)

    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=4)

    vocab = set()
    for entry in processed_data:
        vocab.update(entry['instruction'].lower().split())

    vocab = sorted(vocab)
    with open(vocab_file, 'w') as f:
        json.dump({"vocab": vocab}, f, indent=4)

if __name__ == '__main__':
    preprocess_data()
