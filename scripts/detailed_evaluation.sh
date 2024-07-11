#!/bin/bash

# Script para realizar uma avaliação detalhada e gerar um CSV com os resultados

# Configurar a variável de ambiente PYTHONPATH
export PYTHONPATH=src

# Executar o script de avaliação detalhada
python3 src/training/detailed_evaluation.py
