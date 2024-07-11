#!/bin/bash

# Script para treinar o modelo

# Configurar a variável de ambiente PYTHONPATH
export PYTHONPATH=src

# Executar o script de treinamento
python3 src/training/train.py
