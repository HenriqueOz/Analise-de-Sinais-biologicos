# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 20:41:54 2020

@author: Henrique Oliveira dos Santos
Aluno especial no mestrado de engenharia elétrica da UFSCar
Matéria: Análise de sinais biológicos
Aula: Semana 02
Exercício: Capítulo 12 - Código 2 - DFT direta complexa
COMPLEX DFT BY CORRELATION
Upon entry, N% contains the number of points in the DFT, and XR[ ] and XI[ ] contain the real and imaginary parts of the time domain. Upon return, REX[ ] and IMX[ ] contain the frequency domain data. All signals run from 0 to N%-1.
"""
import numpy as np
import time
import matplotlib.pyplot as plt

def negfreq(ReX, ImX, N):
    M = int((N/2) + 1)
    for k in range(M,N):
        ReX[k] = ReX[N-k+1];
        ImX[k] = -ImX[N-k+1];
    return ReX, ImX

N = 512
print("Com %s amostras" % N)
X1 = np.zeros(N-1)

REX = np.zeros(N-1)
IMX = np.zeros(N-1)
pi = np.pi

# Função mítica # 2*np.sin(np.arange(0,6,0.1))
X1[0:60] = 2*np.sin(np.arange(0,6,0.1)) #[0:int(N/2)] = np.random.randn(int(N/2))

start_time = time.time()
REIMX = np.fft.fft(X1, n =N)
print("FFT real nativa: %s segundos" % (time.time() - start_time))

XR,XI = negfreq(X1,  np.zeros(N-1), int(N/2-1))

# %% Correlate XX[ ] with the cosine and sine waves, Eq. 8-4
start_time = time.time()
for k in range(0,N-1):       #Loop for each value in frequency domain
    for i in range(0, N-1):         #Loop for each value in frequency domain
        SR = np.cos(2*pi*k*i/N)
        SI = -1*np.sin(2*pi*k*i/N)
        REX[k] = REX[k] + XR[i]*SR - XI[i]*SI
        IMX[k] = IMX[k] + XR[i]*SI + XI[i]*SR
print("Método do livro: %s segundos" % (time.time() - start_time))

# %% Função nativa do numpy
start_time = time.time()
REIMX = np.fft.fft(X1, n =N)

plt.plot(REX, marker='o', label = "DSPGuide Method 1")
plt.plot(REIMX.real, label = 'python')
plt.legend()
plt.title("Direct DFT - Parte Real")
plt.xlabel("Frequency Sample number")
plt.ylabel("Amplitude")
plt.figure()
plt.title("Direct DFT - Erro da Parte Real")
plt.plot(np.divide(REX,REIMX.real[0:N-1]))

plt.figure()
plt.plot(IMX, marker='o', label = "DSPGuide Method 1")
plt.plot(REIMX.imag, label = 'python')
plt.legend()
plt.title("Direct DFT - Parte Imaginária")
plt.xlabel("Frequency Sample number")
plt.ylabel("Amplitude")
plt.figure()
plt.plot(np.divide(IMX,REIMX.imag[0:N-1]))
plt.title("Direct DFT - Erro da Parte Imaginária")
