# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 21:34:33 2020

@author: Henrique Oliveira dos Santos
Aluno especial no mestrado de engenharia elétrica da UFSCar
Matéria: Análise de sinais biológicos
Aula: Semana 02
Exercício: Capítulo 8 - Código 2 - DFT direta
"""
import numpy as np
import time
import matplotlib.pyplot as plt

""""
THE DISCRETE FOURIER TRANSFORM
The frequency domain signals, held in REX[ ] and IMX[ ], are calculated from the time domain signal, held in XX[ ].
"""

N = 512
print("Com %s amostras" % N)
X1 = np.zeros(N-1)

REX = np.zeros(int(N/2)+1)
IMX = np.zeros(int(N/2)+1)
pi = np.pi

# Função mítica # 2*np.sin(np.arange(0,6,0.1))
X1[0:60] = 2*np.sin(np.arange(0,6,0.1)) #[0:int(N/2)] = np.random.randn(int(N/2))
# %% Correlate XX[ ] with the cosine and sine waves, Eq. 8-4
start_time = time.time()

for k in range(0,int(N/2)+1):
    for i in range(0, N-1):
        REX[k] = REX[k] + X1[i]*np.cos(2*pi*k*i/N)
        IMX[k] = IMX[k] - X1[i]*np.sin(2*pi*k*i/N)
print("Método do livro: %s segundos" % (time.time() - start_time))

# %% Função nativa do numpy
start_time = time.time()
REIMX = np.fft.rfft(X1, n =N)
print("FFT real nativa: %s segundos" % (time.time() - start_time))

plt.plot(REX, marker='o', label = "DSPGuide Method 1")
plt.plot(REIMX.real, label = 'python')
plt.legend()
plt.title("Direct DFT - Parte Real")
plt.xlabel("Frequency Sample number")
plt.ylabel("Amplitude")
plt.figure()
plt.title("Direct DFT - Erro da Parte Real")
plt.plot(np.divide(REX,REIMX.real))

plt.figure()
plt.plot(IMX, marker='o', label = "DSPGuide Method 1")
plt.plot(REIMX.imag, label = 'python')
plt.legend()
plt.title("Direct DFT - Parte Imaginária")
plt.xlabel("Frequency Sample number")
plt.ylabel("Amplitude")
plt.figure()
plt.plot(np.divide(IMX,REIMX.imag))
plt.title("Direct DFT - Erro da Parte Imaginária")
