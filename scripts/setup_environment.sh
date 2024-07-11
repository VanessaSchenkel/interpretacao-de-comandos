#!/bin/bash

# Script para configurar o ambiente de execução

# Ativar o ambiente virtual (assumindo que você está usando conda)
source ~/anaconda3/etc/profile.d/conda.sh
conda activate trad2

# Instalar as dependências necessárias
pip install -r requirements.txt
