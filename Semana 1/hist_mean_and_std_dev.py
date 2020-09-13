# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 19:33:44 2020

@author: Henrique Oliveira dos Santos
"""
import numpy as np
import time
start_time = time.time()


# signal contém o sinal a ser analisado
n_pontos = 100000
signal = np.zeros(n_pontos)
signal = np.random.randint(low = 0, high = 255, size = n_pontos)

# histogram contém o histograma. Iniciado com zeros
histogram = np.zeros(256)

# Cálculo do histograma para 25001 pontos de valores inteiros
for data_point in signal:
    #H%[ X%[I%] ] = H%[ X%[I%] ] + 1
    histogram[data_point] = histogram[data_point] + 1

# Média calculada via eq 2-6
mean = 0 
for i in range(0,256):
    mean += i*histogram[i]

mean = mean/n_pontos
print("Média pelo algoritmo do livro: ")
print(mean)
print("Média por numpy.average: ")
print(np.mean(signal))

# Cálculo da variancia via eq 2-7
variance = 0
for i in range(0,256):
    variance += histogram[i]*((i-mean)**2)

variance = variance/(n_pontos - 1)
#print("Variância: ")
#print(variance)

std_dev = np.sqrt(variance)
print("Desvio padrão pelo algoritmo do livro:")
print(std_dev)
print("Desvio padrão por numpy.std:")
print(np.std(signal))
print("--- %s seconds ---" % (time.time() - start_time))
