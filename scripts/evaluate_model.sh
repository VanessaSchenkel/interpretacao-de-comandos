#!/bin/bash

# Script para avaliar o modelo

# Configurar a variável de ambiente PYTHONPATH
export PYTHONPATH=src

# Executar o script de avaliação
python3 src/training/evaluate.py
