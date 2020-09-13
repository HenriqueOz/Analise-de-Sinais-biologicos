# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 22:35:36 2020

@author: Henrique Oliveira dos Santos
Aluno especial no mestrado de engenharia elétrica da UFSCar
Matéria: Análise de sinais biológicos
Aula: Semana 02
Exercício: Capítulo 6 - Código 1 - Convulução
"""
import numpy as np
import time
import matplotlib.pyplot as plt

start_time = time.time()
#Criar variáveis a serem utilizadas no exemplo e carregar com dados míticos

np.random.seed(seed = 0)
inputSignal = np.sin(np.linspace(start = 0, stop = 9, num = 100))
impulse = np.array([0, 0, -1, 0, 0])
samples = np.arange(start = 0, stop = impulse.shape[0]+inputSignal.shape[0], step = 1)

#Alocar espaço para a saída e zerá-lo
output = np.zeros(impulse.shape[0]+inputSignal.shape[0]-1)

for i in np.arange(start = 0, stop = inputSignal.shape[0]-1, step = 1):
    for j in np.arange(start = 0, stop = impulse.shape[0]-1, step = 1):
        #print(i)
        #print(j)
        output[i+j] = output[i+j] + inputSignal[i]*impulse[j]

print("--- %s seconds ---" % (time.time() - start_time))

plt.figure(0)
plt.plot(samples[0:output.shape[0]],
         output,
         label='output',
         linestyle = 'dashed',
         marker='o',
         linewidth = 1)
plt.legend()
plt.title("Sinal de saída")
plt.xlabel("Sample number")
plt.ylabel("Amplitude")

plt.figure(1)
plt.plot(samples[0:inputSignal.shape[0]],
         inputSignal,
         label='inputSignal',
         linestyle = 'dashed',
         marker='o',
         linewidth = 1)
plt.legend()
plt.title("Sinal de entrada")
plt.xlabel("Sample number")
plt.ylabel("Amplitude")

plt.figure(2)
plt.plot(samples[0:impulse.shape[0]],
         impulse,
         label='impulse',
         linestyle = 'dashed',
         marker='o',
         linewidth = 1)
plt.legend()
plt.title("Função impulso")
plt.xlabel("Sample number")
plt.ylabel("Amplitude")