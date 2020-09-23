# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 16:26:41 2020

@author: Henrique Oliveira dos Santos
Aluno especial no mestrado de engenharia elétrica da UFSCar
Matéria: Análise de sinais biológicos
Aula: Semana 02
Exercício: Capítulo 7 - Código 1 - Difference equations
"""

# -*- coding: utf-8 -*-
import numpy as np
import time
import matplotlib.pyplot as plt

#Criar variáveis a serem utilizadas no exemplo e carregar com dados míticos

np.random.seed(seed = 0)
inputSignal = np.sin(np.linspace(start = 1, stop = 9, num = 10000))
outputSignal = np.zeros(inputSignal.shape[0])
samples = np.arange(start = 0, stop = inputSignal.shape[0], step = 1)

# %% Primeira derivada
start_time = time.time()

for i in np.arange(start = 0, stop = inputSignal.shape[0]-1, step = 1):
    outputSignal[i] = inputSignal[i] - inputSignal[i-1]
print("Primeira derivada pelo código do livro: %s segundos" % (time.time() - start_time))

start_time = time.time()
outputSignal2 = np.diff(inputSignal)
print("Pelo código do livro: %s segundos" % (time.time() - start_time))

plt.figure(0)
plt.plot(samples,
         inputSignal,
         label='Sinal de entrada',
         linestyle = 'dashed',
         marker='o',
         linewidth = 1)
plt.plot(samples,
         outputSignal,
         label='Sinal de saída derivado',
         linestyle = 'dashed',
         marker='o',
         linewidth = 1)
plt.plot(samples[0:9999],
         outputSignal2,
         label='Sinal de saída pela nativa',
         linestyle = 'solid',
         linewidth = 1)

plt.legend()
plt.title("Derivada discreta ou primeira derivada")
plt.xlabel("Sample number")
plt.ylabel("Amplitude")
# %% Soma corrente
start_time = time.time()

outputSignal = np.zeros(inputSignal.shape[0])
for i in np.arange(start = 0, stop = inputSignal.shape[0]-1, step = 1):
    outputSignal[i] = inputSignal[i] + outputSignal[i-1]

print("Soma cumulativa pelo código do livro: %s segundos" % (time.time() - start_time))

start_time = time.time()
outputSignal2 = np.cumsum(inputSignal)
print("Soma cumulativa pela função nativa: %s segundos" % (time.time() - start_time))

plt.figure(1)
plt.plot(samples,
         inputSignal,
         label='Sinal de entrada',
         linestyle = 'dashed',
         marker='o',
         linewidth = 1)
plt.plot(samples,
         outputSignal,
         label='Sinal de saída integrado',
         linestyle = 'dashed',
         marker='o',
         linewidth = 1)
plt.plot(samples,
         outputSignal2,
         label='Sinal de saída pela nativa',
         linestyle = 'solid',
         linewidth = 1)
plt.legend()
plt.title("Integral discreta ou soma corrente")
plt.xlabel("Sample number")
plt.ylabel("Amplitude")