# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 20:28:05 2020

@author: Henrique Oliveira dos Santos
Aluno especial no mestrado de engenharia elétrica da UFSCar
Matéria: Análise de sinais biológicos
Aula: Semana 02
Exercício: Capítulo 12 - Código 1 - Negative Frequency Generation
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

REX = np.zeros(int(N/2)+1)
IMX = np.zeros(int(N/2)+1)
pi = np.pi

# Função mítica # 2*np.sin(np.arange(0,6,0.1))
X1[0:60] = 2*np.sin(np.arange(0,6,0.1)) #[0:int(N/2)] = np.random.randn(int(N/2))
REIMX = np.fft.rfft(X1, n =N)

REIMX_NEG = negfreq(REIMX.real, REIMX.imag, int(N/2-1))