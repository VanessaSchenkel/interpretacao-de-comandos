import json
import torch
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

def load_training_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return [(entry['instruction'], entry['initial_state'], entry['final_state']) for entry in data]

def encode_state(state, state_mapping):
    encoded = []
    for key in state_mapping:
        encoded.append(state_mapping[key].index(state[key]))
    return encoded

def preprocess_instructions(instructions, vocab, max_length=5):
    tokenized_instructions = []
    for instruction in instructions:
        tokenized_instruction = [vocab.get(word, vocab['<UNK>']) for word in instruction.split()]
        if len(tokenized_instruction) < max_length:
            tokenized_instruction.extend([vocab['<PAD>']] * (max_length - len(tokenized_instruction)))
        else:
            tokenized_instruction = tokenized_instruction[:max_length]
        tokenized_instructions.append(tokenized_instruction)
    return torch.tensor(tokenized_instructions, dtype=torch.long)
