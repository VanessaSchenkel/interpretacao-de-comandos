# Interpretação de Comandos com Tradutor

Este projeto implementa um modelo de aprendizado profundo para interpretar comandos de linguagem natural e executar ações correspondentes em um ambiente simulado.

## Estrutura do Projeto

- `src/`: Contém o código fonte do projeto.
  - `models/`: Contém os modelos de aprendizado profundo.
  - `utils/`: Funções utilitárias para carregamento de dados, pré-processamento, etc.
  - `training/`: Scripts para treinamento e avaliação dos modelos.
- `data/`: Contém os dados de treinamento e teste.
- `scripts/`: Contém scripts úteis para gerenciamento do projeto.

## Dependências

Este projeto utiliza Python 3.11 e várias bibliotecas. Todas as dependências podem ser instaladas usando o arquivo `requirements.txt`.

### Criando o ambiente virtual

É recomendado usar um ambiente virtual para gerenciar as dependências do projeto. Você pode usar o `conda` para criar e ativar o ambiente virtual.

```bash
conda create --name trad2 python=3.11
conda activate trad2
```

### Instalando as dependências

```bash
pip install -r requirements.txt
```

### Treinamento do Modelo

Para treinar o modelo, execute o script train.py localizado no diretório src/training/.

```bash
PYTHONPATH=src python3 src/training/train.py
```

### Avaliação do Modelo

Para avaliar o modelo, execute o script evaluate.py localizado no diretório src/training/.

```bash
PYTHONPATH=src python3 src/training/evaluate.py
```

### Avaliação Detalhada do Modelo

Para realizar uma avaliação detalhada do modelo e gerar logs adicionais, execute o script detailed_evaluation.py localizado no diretório src/training/.

```bash
PYTHONPATH=src python3 src/training/detailed_evaluation.py
```
