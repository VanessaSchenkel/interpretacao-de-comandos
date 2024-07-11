import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from utils.helpers import load_training_data, encode_state, preprocess_instructions
from models.translator import InstructionFollowingModel

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

    training_data = load_training_data('data/training_examples.json')
    instructions, initial_states, final_states = zip(*training_data)

    tokenized_instructions = preprocess_instructions(instructions, vocab, max_length=5)
    encoded_initial_states = torch.tensor(
        [encode_state(state, state_mapping) for state in initial_states],
        dtype=torch.float32
    )
    encoded_final_states = torch.tensor(
        [encode_state(state, state_mapping) for state in final_states],
        dtype=torch.float32
    )

    dataset = TensorDataset(tokenized_instructions, encoded_initial_states, encoded_final_states)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    input_size = len(state_mapping)
    hidden_size = 256
    output_size = len(state_mapping)
    vocab_size = len(vocab)

    model = InstructionFollowingModel(input_size, hidden_size, output_size, vocab_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    num_epochs = 100
    best_val_loss = float('inf')
    patience = 10
    trigger_times = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for instructions, initial_states, final_states in train_loader:
            optimizer.zero_grad()
            outputs = model(instructions, initial_states)
            loss = criterion(outputs, final_states)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * instructions.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for instructions, initial_states, final_states in val_loader:
                outputs = model(instructions, initial_states)
                loss = criterion(outputs, final_states)
                val_loss += loss.item() * instructions.size(0)

        val_loss /= len(val_loader.dataset)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'model-new.pth')
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping')
                break

if __name__ == '__main__':
    main()
