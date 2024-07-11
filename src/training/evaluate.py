import torch
from torch.utils.data import DataLoader, TensorDataset
from utils.helpers import load_training_data, encode_state, preprocess_instructions
from models.translator import InstructionFollowingModel

def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for instructions, initial_states, final_states in data_loader:
            outputs = model(instructions, initial_states)
            _, predicted = torch.max(outputs.data, 1)
            _, target = torch.max(final_states.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def main():
    state_mapping = {
        'microondas': ['fechado', 'aberto'],
        'chaleira': ['fora do fogão', 'no fogão'],
        'luz': ['desligada', 'ligada'],
        'armario': ['fechado', 'aberto'],
        'queimador_superior': ['desligado', 'ligado'],
        'queimador_inferior': ['desligado', 'ligado']
    }

    vocab = {
        'abra': 0, 'a': 1, 'porta': 2, 'do': 3, 'microondas.': 4,
        'coloque': 5, 'chaleira': 6, 'no': 7, 'fogão.': 8,
        'ligue': 9, 'o': 10, 'interruptor': 11, 'da': 12, 'luz.': 13,
        'armário': 14, 'deslizante.': 15,
        '<UNK>': 16, '<PAD>': 17
    }

    testing_data = load_training_data('data/testing_examples.json')
    instructions, initial_states, final_states = zip(*testing_data)

    tokenized_instructions = preprocess_instructions(instructions, vocab, max_length=5)

    encoded_initial_states = torch.tensor(
        [encode_state(state if isinstance(state, dict) else eval(state), state_mapping) for state in initial_states],
        dtype=torch.float32
    )
    encoded_final_states = torch.tensor(
        [encode_state(state if isinstance(state, dict) else eval(state), state_mapping) for state in final_states],
        dtype=torch.float32
    )

    dataset = TensorDataset(tokenized_instructions, encoded_initial_states, encoded_final_states)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    input_size = len(state_mapping)
    hidden_size = 256
    output_size = len(state_mapping)
    vocab_size = len(vocab)

    model = InstructionFollowingModel(input_size, hidden_size, output_size, vocab_size)
    model.load_state_dict(torch.load('model-new.pth'))

    accuracy = evaluate_model(model, data_loader)
    print(f'Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    main()
