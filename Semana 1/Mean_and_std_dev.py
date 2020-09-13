# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 19:33:44 2020

@author: Henrique Oliveira dos Santos
"""

import numpy as np
import time
start_time = time.time()

# Criar vetor para armazenar o dado aquisitado
n_points = 100000
signal = np.zeros(n_points)
# Aquisitar o dado em si
signal = np.random.rand(n_points)

# Variáveis utilizadas ao decorrer dos cálculos

n = 0
sum_ = 0
sumsquares = 0

# Para todo dado presente no vetor, faça a seguinte operação
for data_point in signal:
    n += 1
    sum_ = sum_ + data_point
    sumsquares += data_point**2
    mean = sum_/n
# Caso seja a primeira amostra, std_dev = 0
    if n == 1:
        std_dev = 0
    else:
        std_dev = np.sqrt((sumsquares-sum_**2/n)/(n-1))

print("Desvio padrão pelo algoritmo do livro:")
print(std_dev)
print("Desvio padrão pela biblioteca do numpy:")
print(np.std(signal))
print("--- %s seconds ---" % (time.time() - start_time))
