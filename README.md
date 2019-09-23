# Deep learning para motoristas

Esse projeto tem como objetivo detectar atitudes perigosa enquanto dirige,baseada nas imagens capturadas no painel do carro, usando deep learning.

# Dataset

Os dataset foram obtidos de:

https://www.kaggle.com/c/state-farm-distracted-driver-detection/data

O dataset contem 22,424 imagens que pertence ha uma de 10 classes dadas abaixo:

    c0: Dirigindo com segurança

    c1: mandando mensagem - direita

    c2: falando no celular - direita

    c3: mandando mensagem - esquerda

    c4: falando no celular - esquerda

    c5: mexendo no radio

    c6: bebendo

    c7: checando banco de tras

    c8: mexendo no cabelo ou se arrumando

    c9: falando com passageiro

Os dados foram divididos em dois sets: training set contem 20,924 imagens e o set de validação contem 1500 imagens.

# Pequena Convolutional Neural Network (CNN)

CNN esta implementada no arquivo "driver_distraido.py". Nossa pequena CNN consiste de 3 convolutional layers com o tamanho de filtro 3x3, cada um dos quais é seguido por uma camada de pool máximo (max-pooling layer) com tamanho de pool de 2x2, e 2 camadas densas totalmente conectadas. (fully-connected dense layers).
