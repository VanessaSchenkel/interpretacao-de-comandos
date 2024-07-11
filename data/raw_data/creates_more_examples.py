import json
import random

state_mapping = {
    'microondas': ['fechado', 'aberto'],
    'chaleira': ['fora do fogão', 'no fogão'],
    'luz': ['desligada', 'ligada'],
    'armario': ['fechado', 'aberto'],
    'queimador_superior': ['desligado', 'ligado'],
    'queimador_inferior': ['desligado', 'ligado']
}

instructions = [
    ("Abra a porta do microondas.", {"microondas": "fechado"}, {"microondas": "aberto"}),
    ("Feche a porta do microondas.", {"microondas": "aberto"}, {"microondas": "fechado"}),
    ("Coloque a chaleira no fogão.", {"chaleira": "fora do fogão"}, {"chaleira": "no fogão"}),
    ("Tire a chaleira do fogão.", {"chaleira": "no fogão"}, {"chaleira": "fora do fogão"}),
    ("Ligue a luz.", {"luz": "desligada"}, {"luz": "ligada"}),
    ("Desligue a luz.", {"luz": "ligada"}, {"luz": "desligada"}),
    ("Abra o armário.", {"armario": "fechado"}, {"armario": "aberto"}),
    ("Feche o armário.", {"armario": "aberto"}, {"armario": "fechado"}),
    ("Ligue o queimador superior.", {"queimador_superior": "desligado"}, {"queimador_superior": "ligado"}),
    ("Desligue o queimador superior.", {"queimador_superior": "ligado"}, {"queimador_superior": "desligado"}),
    ("Ligue o queimador inferior.", {"queimador_inferior": "desligado"}, {"queimador_inferior": "ligado"}),
    ("Desligue o queimador inferior.", {"queimador_inferior": "ligado"}, {"queimador_inferior": "desligado"}),
]

def generate_examples(existing_data, num_examples):
    new_examples = []
    while len(new_examples) < num_examples:
        instruction, initial_state_changes, final_state_changes = random.choice(instructions)
        initial_state = random.choice(existing_data)['initial_state'].copy()
        final_state = initial_state.copy()
        
        for key, value in initial_state_changes.items():
            initial_state[key] = value
        for key, value in final_state_changes.items():
            final_state[key] = value

        new_example = {
            "initial_state": initial_state,
            "final_state": final_state,
            "instruction": instruction
        }
        if new_example not in existing_data and new_example not in new_examples:
            new_examples.append(new_example)
    return new_examples

# Carregar dados de treinamento existentes
with open('data/training_examples.json', 'r') as f:
    training_data = json.load(f)

# Gerar novos exemplos
num_new_examples = 30
new_training_examples = generate_examples(training_data, num_new_examples)

# Adicionar novos exemplos aos dados existentes
training_data.extend(new_training_examples)

# Salvar dados de treinamento atualizados
with open('data/training_examples.json', 'w') as f:
    json.dump(training_data, f, indent=4)

print(f"Added {num_new_examples} new examples to training data.")
