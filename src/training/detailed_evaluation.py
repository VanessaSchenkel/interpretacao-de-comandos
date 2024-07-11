import torch
from torch.utils.data import DataLoader, TensorDataset
from utils.helpers import load_training_data, encode_state, preprocess_instructions
from models.translator import InstructionFollowingModel
import pandas as pd

def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    incorrect_predictions = []
    instruction_list = []
    init_state_list = []
    target_list = []
    predicted_list = []

    with torch.no_grad():
        for i, (instructions, initial_states, final_states) in enumerate(data_loader):
            outputs = model(instructions, initial_states)
            _, predicted = torch.max(outputs.data, 1)
            _, target = torch.max(final_states.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            for pred, targ, instr, init_state in zip(predicted, target, instructions, initial_states):
                if pred != targ:
                    incorrect_predictions.append((pred.item(), targ.item()))
                instruction_list.append(instr.tolist())
                init_state_list.append(init_state.tolist())
                target_list.append(targ.item())
                predicted_list.append(pred.item())

            if i % 10 == 0:
                print(f'Batch [{i+1}/{len(data_loader)}], Predicted: {predicted.tolist()}, Target: {target.tolist()}, Correct: {(predicted == target).sum().item()}')

    accuracy = 100 * correct / total
    return accuracy, incorrect_predictions, instruction_list, init_state_list, target_list, predicted_list

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
        '<UNK>': 16, '<PAD>': 17,
        'desligue': 18, 'feche': 19, 'porta.': 20,
        'armário.': 21, 'superior.': 22, 'inferior.': 23, 
        'retire': 24, 'queimador': 25, 'tire': 26,
        'fora': 27, 'do': 28, 'fogão': 29
    }

    print("Carregando dados de teste...")
    testing_data = load_training_data('data/testing_examples.json')
    instructions, initial_states, final_states = zip(*testing_data)

    print("Pré-processando instruções...")
    tokenized_instructions = preprocess_instructions(instructions, vocab, max_length=5)

    print("Codificando estados iniciais e finais...")
    encoded_initial_states = torch.tensor(
        [encode_state(state, state_mapping) for state in initial_states],
        dtype=torch.float32
    )
    encoded_final_states = torch.tensor(
        [encode_state(state, state_mapping) for state in final_states],
        dtype=torch.float32
    )

    print("Criando DataLoader...")
    dataset = TensorDataset(tokenized_instructions, encoded_initial_states, encoded_final_states)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    input_size = len(state_mapping)
    hidden_size = 256
    output_size = len(state_mapping)
    vocab_size = len(vocab)

    print(f"Inicializando modelo com vocab_size={vocab_size}, input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}...")
    model = InstructionFollowingModel(input_size, hidden_size, output_size, vocab_size)

    print("Carregando pesos do modelo pré-treinado...")
    pretrained_model = torch.load('model.pth')

    print(f"Modelo pré-treinado tem embedding de tamanho: {pretrained_model['embedding.weight'].shape}")
    print(f"Modelo atual tem embedding de tamanho: {model.embedding.weight.shape}")

    pretrained_embedding = pretrained_model['embedding.weight']
    model.embedding.weight.data[:pretrained_embedding.shape[0]] = pretrained_embedding

    if pretrained_embedding.shape[0] < vocab_size:
        model.embedding.weight.data[pretrained_embedding.shape[0]:] = torch.randn(vocab_size - pretrained_embedding.shape[0], hidden_size)
        print("Pesos de embedding adicionais inicializados.")

    pretrained_model.pop('embedding.weight')

    print("Carregando estado do modelo...")
    model.load_state_dict(pretrained_model, strict=False)
    print("Estado do modelo carregado com sucesso.")

    accuracy, incorrect_predictions, instruction_list, init_state_list, target_list, predicted_list = evaluate_model(model, data_loader)
    
    df = pd.DataFrame({
        'instruction': instruction_list,
        'initial_state': init_state_list,
        'target': target_list,
        'predicted': predicted_list
    })
    df.to_csv('detailed_evaluation_results.csv', index=False)
    
    print(f'Accuracy: {accuracy:.2f}%')
    print(f'Total de exemplos incorretos: {len(incorrect_predictions)}')
    print(f'Exemplos incorretos (Primeiros 10): {incorrect_predictions[:10]}')

if __name__ == '__main__':
    main()
