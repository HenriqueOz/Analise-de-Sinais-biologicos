# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 20:10:42 2020

@author: Henrique Oliveira dos Santos
Aluno especial no mestrado de engenharia elétrica da UFSCar
Matéria: Análise de sinais biológicos
Aula: Semana 02
Exercício: Capítulo 8 - Código 1 - DFT Inversa
"""

# -*- coding: utf-8 -*-
import numpy as np
import time
import matplotlib.pyplot as plt

""""
THE INVERSE DISCRETE FOURIER TRANSFORM
The time domain signal, held in XX[ ], is calculated from the frequency domain signals, held in REX[ ] and IMX[ ].
"""

N = 4096
print("Com %s amostras" % N)
X1 = np.zeros(N-1)
X2 = np.zeros(N-1)
X3 = np.zeros(N-1)

REX = np.zeros(int(N/2))
IMX = np.zeros(int(N/2))
pi = np.pi

# Função mítica
REX[0:10] = np.random.randn(10) #[0:int(N/2)] = np.random.randn(int(N/2))
ORIGREX = REX
# IMX[0:10] = np.random.randn(10)

REX = np.divide(REX,N/2)
IMX = np.divide(IMX,-N/2)

REX[0] = REX[0]/2
REX[int(N/2 - 1)] = REX[int(N/2 - 1)]

# %% Método de síntese #1 - Correr pelas frequencias
start_time = time.time()
for k in range(0,int(N/2)):
    for i in range(0, N-1):
        X1[i] = X1[i] + REX[k]*np.cos(2*pi*k*i/N)
        X1[i] = X1[i] + IMX[k]*np.sin(2*pi*k*i/N)
print("Método 1: %s segundos" % (time.time() - start_time))

# %% Método de síntese #2 - Correr pelo vetor no tempo
start_time = time.time()

for i in range(0,N-1):
    for k in range(0,int(N/2)):
        X2[i] = X2[i] + REX[k]*np.cos(2*pi*k*i/N)
        X2[i] = X2[i] + IMX[k]*np.sin(2*pi*k*i/N)
print("Método 2: %s segundos" % (time.time() - start_time))

# %% Função nativa do numpy
REIMX = np.empty(REX.shape[0], dtype=complex)
REIMX.real = ORIGREX
REIMX.imag= IMX
# XXX = np.fft.ifft(REIMX, n = int(N))
start_time = time.time()
X3 = np.fft.irfft(REIMX, n = N)
print("Transformada real inversa nativa: %s segundos" % (time.time() - start_time))

plt.plot(X1, label = "DSPGuide Method 1")
plt.plot(X2, label = "DSPGuide Method 2")
plt.plot(X3.real, label = 'python')
plt.legend()
plt.title("Inverse DFT")
plt.xlabel("samples")
plt.ylabel("Amplitude")
plt.figure()
plt.plot(np.divide(X1,X3[0:N-1].real))
plt.title("Inverse DFT - Error between python and book method")
